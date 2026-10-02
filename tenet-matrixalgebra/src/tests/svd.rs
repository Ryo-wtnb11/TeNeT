//! SVD factorization tests (#1596 split of tests.rs).

use super::*;

fn assert_compact_svd_direct_copy_probe() {
    let probe = crate::factorize::compact_svd_copy_probe();
    assert_eq!(probe.input_pack_calls, 0);
    assert_eq!(probe.input_pack_bytes, 0);
    assert_eq!(probe.output_scatter_calls, 0);
    assert_eq!(probe.output_scatter_bytes, 0);
}

#[test]
fn compact_svd_canonical_layout_skips_input_pack_and_factor_scatter() {
    // What: canonical coupled storage reaches final factor destinations without numerical copies.
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    crate::factorize::reset_compact_svd_copy_probe();
    svd_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();

    assert_compact_svd_direct_copy_probe();
}

#[test]
fn compact_svd_noncanonical_layout_uses_copy_fallback() {
    // What: an expert noncanonical view retains the general pack-and-scatter implementation.
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let adjoint_space = bound.space().adjoint_view().unwrap();
    let input = BoundDynamicTensorRef::try_new(&adjoint_space, bound.data()).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    crate::factorize::reset_compact_svd_copy_probe();
    svd_compact_dyn(&mut dense, &input).unwrap();
    let probe = crate::factorize::compact_svd_copy_probe();

    assert!(probe.input_pack_calls > 0);
    assert!(probe.input_pack_bytes > 0);
    assert!(probe.output_scatter_calls > 0);
    assert!(probe.output_scatter_bytes > 0);
}

#[test]
fn direct_compact_svd_uses_owned_executor_outputs_only() {
    let tensor = rectangular_svd_tensor(3, 2);
    let mut direct = RejectSvdInto::default();
    svd_compact(
        &mut direct,
        &bound_tensor_ref!(Arc::new(Z2FusionRule), &tensor),
    )
    .unwrap();
    assert_eq!(direct.svd_calls, 1);
    assert_eq!(direct.svd_into_calls, 0);

    let (space, data) = generic_factorization_input();
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let mut generic = RejectSvdInto::default();
    svd_compact_factors_dyn_generic(&mut generic, &input).unwrap();
    assert!(generic.svd_calls > 0);
    assert_eq!(generic.svd_into_calls, 0);

    let (_, checked_space) = bind_checked_only(&space);
    let checked_input = BoundDynamicTensorRef::try_new(&checked_space, &data).unwrap();
    let mut checked = RejectSvdInto::default();
    svd_compact_dyn_checked_generic(&mut checked, &checked_input).unwrap();
    assert!(checked.svd_calls > 0);
    assert_eq!(checked.svd_into_calls, 0);

    let bound = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let mut polar = RejectSvdInto::default();
    let mut context = default_context();
    left_polar(&mut polar, &mut context, &bound.as_ref()).unwrap();
    assert_eq!(polar.svd_calls, 1);
    assert_eq!(polar.svd_into_calls, 0);

    let fallback_space = bound.space().adjoint_view().unwrap();
    let fallback = BoundDynamicTensorRef::try_new(&fallback_space, bound.data()).unwrap();
    let mut legacy = RejectSvdInto::default();
    crate::factorize::reset_compact_svd_copy_probe();
    svd_compact_dyn(&mut legacy, &fallback).unwrap();
    assert!(legacy.svd_calls > 0);
    assert_eq!(legacy.svd_into_calls, 0);
    let probe = crate::factorize::compact_svd_copy_probe();
    assert!(probe.input_pack_calls > 0);
    assert!(probe.output_scatter_calls > 0);
}

fn assert_mf_compact_svd_fallback_live_owners<D: crate::factorize::FactorScalar>() {
    let source = mixed_rectangular_c32_tensor();
    let tensor = TensorMap::<D, 1, 1>::from_vec_with_fusion_space(
        source
            .data()
            .iter()
            .map(|value| D::from_complex64(Complex64::new(value.re as f64, value.im as f64)))
            .collect(),
        source.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let bound = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let fallback_space = bound.space().adjoint_view().unwrap();
    let fallback = BoundDynamicTensorRef::try_new(&fallback_space, bound.data()).unwrap();
    let mut dense = RejectSvdInto::default();

    crate::factorize::reset_compact_svd_copy_probe();
    crate::factorize::reset_mf_compact_svd_fallback_pointers();
    svd_compact_dyn(&mut dense, &fallback).unwrap();
    let stage = crate::factorize::mf_compact_svd_fallback_pointers();

    assert_eq!(dense.svd_into_calls, 0);
    assert_eq!(dense.svd_calls, 2);
    assert_eq!(dense.output_ptrs, stage);
    assert_eq!(stage.len(), 2);
    assert!(stage.iter().all(|&(u, vt)| u != 0 && vt != 0 && u != vt));
    let probe = crate::factorize::compact_svd_copy_probe();
    assert!(probe.input_pack_calls > 0);
    assert!(probe.output_scatter_calls > 0);

    let mut adjoint_dense = RejectSvdInto::default();
    crate::factorize::reset_compact_svd_copy_probe();
    crate::factorize::reset_mf_compact_svd_fallback_pointers();
    svd_compact_adjoint_factors_dyn(&mut adjoint_dense, &fallback).unwrap();
    let adjoint_stage = crate::factorize::mf_compact_svd_fallback_pointers();
    assert_eq!(adjoint_dense.svd_into_calls, 0);
    assert_eq!(adjoint_dense.svd_calls, 2);
    assert_eq!(adjoint_dense.output_ptrs, adjoint_stage);
    assert_eq!(adjoint_stage.len(), 2);
    assert!(adjoint_stage
        .iter()
        .all(|&(u, vt)| u != 0 && vt != 0 && u != vt));
    let adjoint_probe = crate::factorize::compact_svd_copy_probe();
    assert!(adjoint_probe.input_pack_calls > 0);
    assert!(adjoint_probe.output_scatter_calls > 0);
}

#[test]
fn mf_compact_svd_fallback_keeps_live_owners_for_every_dtype() {
    assert_mf_compact_svd_fallback_live_owners::<f64>();
    assert_mf_compact_svd_fallback_live_owners::<f32>();
    assert_mf_compact_svd_fallback_live_owners::<Complex32>();
    assert_mf_compact_svd_fallback_live_owners::<Complex64>();
}

#[test]
fn generic_compact_svd_padded_fallback_uses_owned_outputs_and_scatter() {
    let (canonical_space, canonical_data) = generic_factorization_input();
    let (padded_space, padded_data) =
        padded_generic_factorization_input(&canonical_space, &canonical_data);
    let padded = BoundDynamicTensorRef::try_new(&padded_space, &padded_data).unwrap();
    let mut dense = RejectSvdInto::default();

    crate::factorize::reset_compact_svd_copy_probe();
    svd_compact_factors_dyn_generic(&mut dense, &padded).unwrap();

    assert!(dense.svd_calls > 0);
    assert_eq!(dense.svd_into_calls, 0);
    let probe = crate::factorize::compact_svd_copy_probe();
    assert!(probe.input_pack_calls > 0);
    assert!(probe.output_scatter_calls > 0);
}

fn assert_generic_compact_svd_fallback_live_owners<D: crate::factorize::FactorScalar>() {
    let (canonical_space, canonical_data) = generic_factorization_input();
    let (padded_space, padded_data) =
        padded_generic_factorization_input(&canonical_space, &canonical_data);
    let data = padded_data
        .into_iter()
        .map(D::from_real)
        .collect::<Vec<_>>();
    let padded = BoundDynamicTensorRef::try_new(&padded_space, &data).unwrap();
    let mut dense = RejectSvdInto::default();

    crate::factorize::reset_compact_svd_copy_probe();
    crate::factorize::reset_generic_compact_svd_fallback_pointers();
    svd_compact_factors_dyn_generic(&mut dense, &padded).unwrap();
    let stage = crate::factorize::generic_compact_svd_fallback_pointers();

    assert_eq!(dense.svd_into_calls, 0);
    assert_eq!(dense.output_ptrs, stage);
    assert_eq!(stage.len(), 2);
    assert!(stage.iter().all(|&(u, vt)| u != 0 && vt != 0 && u != vt));
    let probe = crate::factorize::compact_svd_copy_probe();
    assert!(probe.input_pack_calls > 0);
    assert!(probe.output_scatter_calls > 0);
}

#[test]
fn generic_compact_svd_fallback_keeps_live_owners_for_every_dtype() {
    assert_generic_compact_svd_fallback_live_owners::<f64>();
    assert_generic_compact_svd_fallback_live_owners::<f32>();
    assert_generic_compact_svd_fallback_live_owners::<Complex32>();
    assert_generic_compact_svd_fallback_live_owners::<Complex64>();
}

#[test]
fn generic_compact_svd_interleaved_complex_fallback_preserves_source_order() {
    let (canonical_space, canonical_data, _) = generic_values_endomorphism_input();
    let (interleaved_space, interleaved_data) =
        interleaved_generic_endomorphism_input(&canonical_space, &canonical_data);
    let canonical = BoundDynamicTensorRef::try_new(&canonical_space, &canonical_data).unwrap();
    let interleaved =
        BoundDynamicTensorRef::try_new(&interleaved_space, &interleaved_data).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let canonical_svd = svd_compact_factors_dyn_generic(&mut dense, &canonical).unwrap();
    crate::factorize::reset_compact_svd_copy_probe();
    let fallback_svd = svd_compact_factors_dyn_generic(&mut dense, &interleaved).unwrap();

    let fallback_s = generic_diagonal_factor(&fallback_svd.0, &fallback_svd.2);
    assert_compact_factors_reconstruct_input(
        &interleaved,
        &fallback_svd.0,
        Some(&fallback_s),
        &fallback_svd.1,
    );
    assert_eq!(
        fallback_svd
            .2
            .iter()
            .map(|entry| entry.sector)
            .collect::<Vec<_>>(),
        [SectorId::new(1), SectorId::new(0)]
    );
    for actual in &fallback_svd.2 {
        let expected = canonical_svd
            .2
            .iter()
            .find(|entry| entry.sector == actual.sector)
            .unwrap();
        assert_eq!(actual.values.len(), expected.values.len());
        for (&actual, &expected) in actual.values.iter().zip(&expected.values) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
    }
    let probe = crate::factorize::compact_svd_copy_probe();
    assert!(probe.input_pack_calls > 0);
    assert!(probe.output_scatter_calls > 0);
}

#[test]
fn generic_compact_svd_padded_complex_rectangular_fallback_matches_canonical_gauge() {
    let (canonical_space, canonical_data) = generic_svd_truncation_input::<Complex64>(true);
    let (padded_space, padded_data) =
        padded_generic_svd_truncation_input(&canonical_space, &canonical_data);
    let canonical = BoundDynamicTensorRef::try_new(&canonical_space, &canonical_data).unwrap();
    let padded = BoundDynamicTensorRef::try_new(&padded_space, &padded_data).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let canonical_svd = svd_compact_factors_dyn_generic(&mut dense, &canonical).unwrap();
    crate::factorize::reset_compact_svd_copy_probe();
    let mut reject = RejectSvdInto::default();
    let padded_svd = svd_compact_factors_dyn_generic(&mut reject, &padded).unwrap();

    assert_eq!(reject.svd_into_calls, 0);
    let padded_s = generic_diagonal_factor(&padded_svd.0, &padded_svd.2);
    assert_compact_factors_reconstruct_input(
        &padded,
        &padded_svd.0,
        Some(&padded_s),
        &padded_svd.1,
    );
    assert_generic_complex_factor_close(&padded_svd.0, &canonical_svd.0);
    assert_generic_complex_factor_close(&padded_svd.1, &canonical_svd.1);
    assert_real_spectra_close(&padded_svd.2, &canonical_svd.2);
    let probe = crate::factorize::compact_svd_copy_probe();
    assert!(probe.input_pack_calls > 0);
    assert!(probe.output_scatter_calls > 0);
}

#[test]
fn generic_compact_svd_second_dense_failure_preserves_source() {
    let (source_space, source_data) = generic_svd_truncation_input::<f64>(false);
    let (space, data) = padded_generic_svd_truncation_input(&source_space, &source_data);
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let before = input.data().to_vec();
    let mut dense = FailSecondSvd::default();

    crate::factorize::reset_compact_svd_copy_probe();
    match svd_compact_factors_dyn_generic(&mut dense, &input) {
        Err(OperationError::Dense(DenseError::Backend { op: "svd_into", .. })) => {}
        Err(error) => panic!("unexpected Generic SVD failure: {error}"),
        Ok(_) => panic!("second compact SVD must fail"),
    }
    assert_eq!(dense.calls, 2);
    assert_eq!(input.data(), before);
    let probe = crate::factorize::compact_svd_copy_probe();
    assert!(probe.input_pack_calls > 0);
    assert!(probe.output_scatter_calls > 0);
}

#[test]
fn generic_compact_svd_empty_input_skips_dense_execution() {
    let x = SectorId::new(1);
    let empty_leg = SectorLeg::new([(x, 0)], false);
    let space = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([empty_leg.clone()]),
            FusionProductSpace::new([empty_leg]),
        ),
    )
    .unwrap();
    let data: [f64; 0] = [];
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let mut reject = RejectExecutorCalls;

    let (u, vh, singular_values) = svd_compact_factors_dyn_generic(&mut reject, &input).unwrap();

    assert!(u.data().is_empty());
    assert!(vh.data().is_empty());
    assert!(singular_values.is_empty());
}

fn assert_checked_compact_svd<D>(complex: bool)
where
    D: FactorScalar,
{
    let kept = 2;
    let (provider, space, data) = checked_svd_truncation_input::<D>(complex);
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let mut dense = CountingDense::default();
    let Svd { u, s, vh } = svd_compact_dyn_checked_generic(&mut dense, &input).unwrap();

    assert_eq!(dense.svd_calls, 2);
    assert_eq!(dense.svd_into_calls, 0);
    assert_eq!(dense.svd_vals_calls, 0);
    assert!(Arc::ptr_eq(u.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(vh.space().provider_arc(), &provider));

    let expected_spectra = [
        (SectorId::new(0), [4.0, 1.0]),
        (SectorId::new(1), [3.0, 2.0]),
    ];
    let mut residual_squared = 0.0;
    for (sector, expected) in expected_spectra {
        // The checked compact S is dense: read its diagonal.
        let s_block = (0..s.space().space().structure().block_count())
            .map(|index| s.space().space().structure().block(index).unwrap())
            .find(|block| {
                matches!(
                    block.key(),
                    BlockKey::FusionTree(key) if key.codomain_tree().coupled() == sector
                )
            })
            .unwrap();
        let spectrum = (0..s_block.shape()[0])
            .map(|k| {
                s.data()[s_block.offset() + k * (s_block.strides()[0] + s_block.strides()[1])]
                    .widen_complex()
                    .re
            })
            .collect::<Vec<f64>>();
        assert_eq!(spectrum.len(), kept);
        for (&actual, &expected) in spectrum.iter().zip(&expected[..kept]) {
            assert!((actual - expected).abs() < 1.0e-10);
        }

        let u_block = (0..u.space().space().structure().block_count())
            .map(|index| u.space().space().structure().block(index).unwrap())
            .find(|block| {
                matches!(
                    block.key(),
                    BlockKey::FusionTree(key) if key.codomain_tree().coupled() == sector
                )
            })
            .unwrap();
        let vh_block = (0..vh.space().space().structure().block_count())
            .map(|index| vh.space().space().structure().block(index).unwrap())
            .find(|block| {
                matches!(
                    block.key(),
                    BlockKey::FusionTree(key) if key.domain_tree().coupled() == sector
                )
            })
            .unwrap();
        let (rows, cols, matrix) = checked_svd_matrix(sector, complex);
        assert_eq!(u_block.shape(), [rows, kept]);
        assert_eq!(vh_block.shape(), [kept, cols]);

        let u_value = |row: usize, col: usize| {
            u.data()[u_block.offset() + row * u_block.strides()[0] + col * u_block.strides()[1]]
                .widen_complex()
        };
        let vh_value = |row: usize, col: usize| {
            vh.data()[vh_block.offset() + row * vh_block.strides()[0] + col * vh_block.strides()[1]]
                .widen_complex()
        };
        for left in 0..kept {
            for right in 0..kept {
                let u_inner = (0..rows)
                    .map(|row| u_value(row, left).conj() * u_value(row, right))
                    .sum::<Complex64>();
                let vh_inner = (0..cols)
                    .map(|col| vh_value(left, col) * vh_value(right, col).conj())
                    .sum::<Complex64>();
                let expected = if left == right { 1.0 } else { 0.0 };
                assert!((u_inner - expected).norm() < 1.0e-10);
                assert!((vh_inner - expected).norm() < 1.0e-10);
            }
        }
        for col in 0..cols {
            for row in 0..rows {
                let reconstructed = (0..kept)
                    .map(|bond| u_value(row, bond) * spectrum[bond] * vh_value(bond, col))
                    .sum::<Complex64>();
                residual_squared += if sector == SectorId::new(1) {
                    (1.0 + 2.0_f64.sqrt()) * (reconstructed - matrix[row + rows * col]).norm_sqr()
                } else {
                    (reconstructed - matrix[row + rows * col]).norm_sqr()
                };
            }
        }
    }
    assert!(residual_squared.sqrt() < 1.0e-10);
}

#[test]
fn checked_generic_svd_compact_uses_each_real_compact_decomposition_once() {
    assert_checked_compact_svd::<f64>(false);
}

#[test]
fn checked_generic_svd_compact_uses_each_complex_compact_decomposition_once() {
    assert_checked_compact_svd::<Complex64>(true);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_svd_compact_empty_input_skips_dense_execution() {
    let vacuum = SectorId::new(0);
    let x = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(vacuum, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(x, 1)], false)]),
    );
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        homspace,
    )
    .unwrap();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), Arc::clone(&provider))
            .unwrap();
    let data = Vec::<f64>::new();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = CountingDense::default();
    let Svd { u, s, vh } = svd_compact_dyn_checked_generic(&mut dense, &input).unwrap();

    assert_eq!(dense.svd_calls, 0);
    assert_eq!(dense.svd_into_calls, 0);
    assert_eq!(dense.svd_vals_calls, 0);
    assert!(s.data().is_empty());
    assert!(u.data().is_empty());
    assert!(vh.data().is_empty());
    assert!(Arc::ptr_eq(u.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(vh.space().provider_arc(), &provider));
}

#[test]
fn checked_generic_svd_compact_dense_failure_precedes_output_provider_admission() {
    let (provider, space, data) = checked_svd_truncation_input::<f64>(false);
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let before = input.data().to_vec();
    let mut dense = FailAfterObservingSvdInput::default();
    let result = svd_compact_dyn_checked_generic(&mut dense, &input);

    assert!(matches!(
        result,
        Err(CheckedGenericFactorPlanError::Operation(
            OperationError::Dense(DenseError::Backend { op: "svd_into", .. })
        ))
    ));
    assert_eq!(provider.calls.get(), 0);
    assert_eq!(dense.observed.len(), 1);
    assert_eq!(input.data(), before);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_full_svd_preserves_provider_and_completes_unmatched_rows() {
    let rule = FactorGenericRule;
    let x = SectorId::new(1);
    let vacuum = SectorId::new(0);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(x, 1), (vacuum, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(x, 1)], false)]),
    );
    let source =
        BoundDynamicFusionMapSpace::from_final_homspace_generic(Arc::new(rule), homspace).unwrap();
    let data = vec![1.0; source.space().required_len().unwrap()];
    let checked_provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let checked = BoundDynamicFusionMapSpace::bind_generic(
        source.space().clone(),
        Arc::clone(&checked_provider),
    )
    .unwrap();
    checked_provider.calls.set(0);
    assert!(matches!(
        svd_full_diagonal_factors_dyn_checked_generic::<_, f64>(&checked, &[]).unwrap(),
        CheckedDiagonalFullSvdFactors::NotAdmitted
    ));
    assert_eq!(checked_provider.calls.get(), 0);
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = CountingDense::default();
    let full = svd_full_dyn_checked_generic(&mut dense, &input).unwrap();
    assert_eq!(dense.svd_calls, 1);
    assert!(Arc::ptr_eq(
        full.u().space().provider_arc(),
        &checked_provider
    ));
    assert!(Arc::ptr_eq(
        full.s().space().provider_arc(),
        &checked_provider
    ));
    assert!(Arc::ptr_eq(
        full.vh().space().provider_arc(),
        &checked_provider
    ));
    let structure = full.u().space().space().structure();
    assert!((0..structure.block_count()).any(|index| {
        matches!(
            structure.block(index).unwrap().key(),
            BlockKey::FusionTree(key) if key.codomain_tree().coupled() == vacuum
        )
    }));
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_compact_diagonal_full_svd_has_no_post_preflight_provider_query() {
    let x = SectorId::new(1);
    let leg = SectorLeg::new([(x, 2)], true);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone()]),
        FusionProductSpace::new([leg]),
    );
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        homspace,
    )
    .unwrap();
    let spectrum = [SectorSpectrum {
        sector: x,
        values: vec![2.0, 1.0],
    }];

    let successful_provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let successful = BoundDynamicFusionMapSpace::bind_generic(
        source.space().clone(),
        Arc::clone(&successful_provider),
    )
    .unwrap();
    successful_provider.calls.set(0);
    coupled_sector_block_dimensions_generic_checked(
        successful.space().homspace().codomain(),
        successful_provider.as_ref(),
    )
    .unwrap();
    coupled_sector_block_dimensions_generic_checked(
        successful.space().homspace().domain(),
        successful_provider.as_ref(),
    )
    .unwrap();
    let dimension_calls = successful_provider.calls.get();
    successful_provider.calls.set(0);
    assert!(matches!(
        svd_full_diagonal_factors_dyn_checked_generic(&successful, &spectrum).unwrap(),
        CheckedDiagonalFullSvdFactors::Direct(_)
    ));
    assert_eq!(successful_provider.calls.get(), dimension_calls);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_full_svd_failure_publishes_no_factors() {
    let (source, data) = generic_factorization_input();
    let failing = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: 2,
        calls: Cell::new(0),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), failing).unwrap();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let result = svd_full_dyn_checked_generic(&mut dense, &input);
    assert!(matches!(
        result,
        Err(crate::CheckedGenericFactorPlanError::Provider(_))
    ));
}

fn assert_checked_compact_svd_live_stage_owners<D: crate::factorize::FactorScalar>() {
    let (_, space, data) = checked_svd_truncation_input::<D>(true);
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let mut dense = RejectSvdInto::default();
    crate::factorize::reset_checked_compact_svd_stage_pointers();

    svd_compact_dyn_checked_generic(&mut dense, &input).unwrap();
    let stage = crate::factorize::checked_compact_svd_stage_pointers();

    assert_eq!(dense.svd_into_calls, 0);
    assert_eq!(dense.svd_calls, 2);
    assert_eq!(stage.len(), 2);
    assert_eq!(dense.output_ptrs, stage);
    assert!(stage.iter().all(|&(u, vt)| u != 0 && vt != 0 && u != vt));
}

#[test]
fn checked_generic_compact_svd_keeps_live_stage_owners_for_every_dtype() {
    assert_checked_compact_svd_live_stage_owners::<f64>();
    assert_checked_compact_svd_live_stage_owners::<f32>();
    assert_checked_compact_svd_live_stage_owners::<Complex32>();
    assert_checked_compact_svd_live_stage_owners::<Complex64>();
}

#[test]
fn checked_compact_svd_zero_rank_skips_backend_and_post_gauge_stage() {
    let mut dense = RejectSvdInto::default();
    crate::factorize::reset_checked_compact_svd_stage_pointers();

    for (rows, cols) in [(0, 3), (3, 0)] {
        assert_eq!(
            crate::factorize::compact_svd_numerical_stage_lengths_for_test(
                &mut dense,
                &[] as &[f64],
                rows,
                cols,
            )
            .unwrap(),
            (0, 0, 0)
        );
    }
    assert_eq!(dense.svd_calls, 0);
    assert_eq!(dense.svd_into_calls, 0);
    assert!(crate::factorize::checked_compact_svd_stage_pointers().is_empty());
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_full_svd_completes_unmatched_columns_and_disjoint_space() {
    let rule = FactorGenericRule;
    let x = SectorId::new(1);
    let vacuum = SectorId::new(0);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(x, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(x, 1), (vacuum, 1)], false)]),
    );
    let source =
        BoundDynamicFusionMapSpace::from_final_homspace_generic(Arc::new(rule), homspace).unwrap();
    let data = vec![1.0; source.space().required_len().unwrap()];
    let checked = BoundDynamicFusionMapSpace::bind_generic(
        source.space().clone(),
        Arc::new(LateGenericSpy {
            rule: FactorGenericRule,
            fail_at: usize::MAX,
            calls: Cell::new(0),
        }),
    )
    .unwrap();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let full = svd_full_dyn_checked_generic(&mut dense, &input).unwrap();
    let structure = full.vh().space().space().structure();
    assert!((0..structure.block_count()).any(|index| {
        matches!(
            structure.block(index).unwrap().key(),
            BlockKey::FusionTree(key) if key.domain_tree().coupled() == vacuum
        )
    }));

    let disjoint = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(vacuum, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(x, 1)], false)]),
    );
    let source =
        BoundDynamicFusionMapSpace::from_final_homspace_generic(Arc::new(rule), disjoint).unwrap();
    let checked = BoundDynamicFusionMapSpace::bind_generic(
        source.space().clone(),
        Arc::new(LateGenericSpy {
            rule: FactorGenericRule,
            fail_at: usize::MAX,
            calls: Cell::new(0),
        }),
    )
    .unwrap();
    let data = vec![1.0; source.space().required_len().unwrap()];
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let full = svd_full_dyn_checked_generic(&mut dense, &input).unwrap();
    assert!(full.singular_values().is_empty());
}

#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn assert_checked_full_svd_builder_failure(fail_at: usize) {
    let (source, data) = generic_factorization_input();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at,
        calls: Cell::new(0),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), Arc::clone(&provider))
            .unwrap();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let before = input.data().to_vec();
    let mut dense = CountingDense::default();
    let result = svd_full_dyn_checked_generic(&mut dense, &input);

    assert!(matches!(
        result,
        Err(crate::CheckedGenericFactorPlanError::Provider(LateGenericError(call)))
            if call == fail_at
    ));
    assert_eq!(input.data(), before);
    assert!(Arc::ptr_eq(input.space().provider_arc(), &provider));
    assert_eq!(dense.svd_calls, 2);
    assert_eq!(dense.qr_calls, 2);
}

// Checked full-SVD provider sequence for `generic_factorization_input`: ten
// calls of the two multiplicity-aware dimension preflights, then the single
// U output-layout enumeration (calls 11-13), then the single Vh enumeration
// (calls 14-16); the diagonal S space issues no provider query. Before the
// one-sided owner enumerated its layout once, each output enumerated twice
// (U 11-16, Vh 17-22) and these tests pinned calls 15, 19 and 22.
const FULL_SVD_DIMENSION_PREFLIGHT_CALLS: usize = 10;
const FULL_SVD_U_FIRST_CALL: usize = FULL_SVD_DIMENSION_PREFLIGHT_CALLS + 1;
const FULL_SVD_U_LAST_CALL: usize = FULL_SVD_DIMENSION_PREFLIGHT_CALLS + 3;
const FULL_SVD_VH_FIRST_CALL: usize = FULL_SVD_U_LAST_CALL + 1;
const FULL_SVD_VH_LAST_CALL: usize = FULL_SVD_U_LAST_CALL + 3;

#[test]
fn checked_generic_full_svd_u_builder_failure_preserves_provider_context() {
    // What: the first and last post-dense checked-provider calls of U-space
    // construction propagate their exact provider error.
    assert_checked_full_svd_builder_failure(FULL_SVD_U_FIRST_CALL);
    assert_checked_full_svd_builder_failure(FULL_SVD_U_LAST_CALL);
}

#[test]
fn checked_generic_full_svd_vh_builder_failure_preserves_provider_context() {
    // What: Vh-space construction propagates its exact provider error without
    // publishing U, from its first call to its last.
    assert_checked_full_svd_builder_failure(FULL_SVD_VH_FIRST_CALL);
    assert_checked_full_svd_builder_failure(FULL_SVD_VH_LAST_CALL);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_full_svd_enumerates_each_output_layout_once() {
    // What: the provider sees the dimension preflights plus exactly one
    // layout enumeration per one-sided output; the diagonal S publication
    // adds none. Formerly 22 calls (each output enumerated twice), now 16.
    let (source, data) = generic_factorization_input();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), Arc::clone(&provider))
            .unwrap();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = CountingDense::default();
    let output = svd_full_dyn_checked_generic(&mut dense, &input).unwrap();
    assert_eq!(provider.calls.get(), FULL_SVD_VH_LAST_CALL);

    let probe_calls = |run: &dyn Fn(&LateGenericSpy)| {
        let probe = LateGenericSpy {
            rule: FactorGenericRule,
            fail_at: usize::MAX,
            calls: Cell::new(0),
        };
        run(&probe);
        probe.calls.get()
    };
    let homspace = source.space().homspace();
    let preflight = probe_calls(&|probe| {
        coupled_sector_block_dimensions_generic_checked(homspace.codomain(), probe).unwrap();
        coupled_sector_block_dimensions_generic_checked(homspace.domain(), probe).unwrap();
    });
    assert_eq!(preflight, FULL_SVD_DIMENSION_PREFLIGHT_CALLS);
    let enumeration = |factor: &BoundDynFactor<LateGenericSpy, f64>| {
        let homspace = factor.space().space().homspace().clone();
        probe_calls(&|probe| {
            homspace
                .prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(probe)
                .unwrap();
        })
    };
    assert_eq!(
        enumeration(output.u()),
        FULL_SVD_U_LAST_CALL - FULL_SVD_U_FIRST_CALL + 1
    );
    assert_eq!(
        enumeration(output.vh()),
        FULL_SVD_VH_LAST_CALL - FULL_SVD_VH_FIRST_CALL + 1
    );
    assert_eq!(enumeration(output.s()), 0);
    assert!(Arc::ptr_eq(output.u().space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(output.vh().space().provider_arc(), &provider));
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_native_full_svd_stages_before_unchanged_provider_admission() {
    let (source, data) = generic_factorization_input();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), Arc::clone(&provider))
            .unwrap();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let Some(mut dense) = NativeFullSvdSpy::new() else {
        return;
    };

    let full = svd_full_dyn_checked_generic(&mut dense, &input).unwrap();

    assert_eq!(dense.full_calls, 2);
    assert_eq!(provider.calls.get(), FULL_SVD_VH_LAST_CALL);
    assert!(Arc::ptr_eq(full.u().space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(full.s().space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(full.vh().space().provider_arc(), &provider));
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_full_svd_local_shape_error_precedes_provider_query() {
    // What: checked tensor admission reports the local storage mismatch before
    // any provider-backed factorization work can run.
    let (source, data) = generic_factorization_input();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: 1,
        calls: Cell::new(0),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), Arc::clone(&provider))
            .unwrap();
    let error = match BoundDynamicTensorRef::try_new(&checked, &data[..data.len() - 1]) {
        Ok(_) => panic!("short storage must be rejected"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        OperationError::Core(CoreError::DimensionMismatch { .. })
    ));
    assert_eq!(provider.calls.get(), 0);
}

#[test]
fn compact_svd_adjoint_error_preserves_borrowed_input_and_publishes_no_factors() {
    // What: an SVD backend failure leaves the parent storage unchanged and
    // returns no partially constructed adjoint factors.
    let rule = Z2FusionRule;
    let tensor = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let before = tensor.data().to_vec();
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let mut adjoint_dense = FailAfterObservingSvdInput::default();
    let result = svd_compact_adjoint_factors_dyn(&mut adjoint_dense, &bound.as_ref().dynamic());
    assert!(matches!(result, Err(OperationError::Dense(_))));
    assert_eq!(tensor.data(), before);
    assert!(!adjoint_dense.observed.is_empty());
}

#[test]
fn compact_svd_adjoint_late_error_preserves_borrowed_input_and_publishes_no_factors() {
    let rule = Z2FusionRule;
    let canonical = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let tensor = padded_copy(&rule, &canonical);
    let before = tensor.data().to_vec();
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let mut dense = FailSecondSvd::default();
    crate::factorize::reset_compact_svd_copy_probe();

    let result = svd_compact_adjoint_factors_dyn(&mut dense, &bound.as_ref().dynamic());

    assert!(matches!(result, Err(OperationError::Dense(_))));
    assert_eq!(tensor.data(), before);
    assert_eq!(dense.calls, 2);
    let probe = crate::factorize::compact_svd_copy_probe();
    assert!(probe.input_pack_calls > 0);
    assert!(probe.output_scatter_calls > 0);
}

#[test]
fn full_svd_late_error_preserves_input_and_publishes_no_factors() {
    // What: the adjoint-oriented full-SVD engine finishes every sector before
    // allocating any returned factor.
    let rule = Z2FusionRule;
    let tensor = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let before = tensor.data().to_vec();
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let mut dense = FailSecondSvd::default();

    crate::factorize::reset_factor_buffer_build_counts_for_test();
    let result = svd_full_adjoint_dyn(&mut dense, &bound.as_ref().dynamic());

    assert!(matches!(result, Err(OperationError::Dense(_))));
    assert_eq!(tensor.data(), before);
    assert_eq!(dense.calls, 2);
    assert_eq!(
        crate::factorize::factor_buffer_build_counts_for_test(),
        (0, 0)
    );
}

#[test]
fn full_svd_publishes_owned_factors_without_scatter_in_both_orientations() {
    // What: the production full SVD (direct and adjoint engines) admits U and
    // Vh layouts that the staged per-sector factors already occupy, so both
    // factors are transferred instead of zero-filled and scattered.
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    crate::factorize::reset_one_sided_publication_probe();
    let direct = svd_full_dyn(&mut dense, &bound.as_ref().dynamic()).unwrap();
    let probe = crate::factorize::one_sided_publication_probe();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (2, 0)
    );
    assert!(probe.appended_elements > 0);

    crate::factorize::reset_one_sided_publication_probe();
    let adjoint = svd_full_adjoint_dyn(&mut dense, &bound.as_ref().dynamic()).unwrap();
    let probe = crate::factorize::one_sided_publication_probe();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (2, 0)
    );
    assert_eq!(
        direct.u().space().space().required_len().unwrap(),
        adjoint.u().space().space().required_len().unwrap()
    );
}

#[test]
fn full_svd_adjoint_builds_only_the_final_factor_buffers() {
    let tensor = one_sector_rectangular_matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0], 2, 3);
    let bound = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    crate::factorize::reset_factor_buffer_build_counts_for_test();
    let output = svd_full_adjoint_dyn(&mut dense, &bound.as_ref().dynamic()).unwrap();

    assert_eq!(
        crate::factorize::factor_buffer_build_counts_for_test(),
        (1, 1)
    );
    assert_eq!(output.u().space().space().required_len().unwrap(), 9);
    assert_eq!(output.s().space().space().required_len().unwrap(), 6);
    assert_eq!(output.vh().space().space().required_len().unwrap(), 4);
}

#[test]
fn full_svd_adjoint_completion_reconstructs_rectangular_input() {
    let tensor = one_sector_rectangular_matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0], 2, 3);
    let bound = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let mut dense = SvdCallSpy::default();
    let output = svd_full_adjoint_dyn(&mut dense, &bound.as_ref().dynamic()).unwrap();
    assert_eq!(dense.svd_calls, 1);
    let u = output.u().data();
    let s = output.s().data();
    let vh = output.vh().data();
    for col in 0..2 {
        for row in 0..3 {
            let reconstructed = (0..3)
                .map(|middle| {
                    (0..2)
                        .map(|inner| {
                            u[row + 3 * middle] * s[middle + 3 * inner] * vh[inner + 2 * col]
                        })
                        .sum::<f64>()
                })
                .sum::<f64>();
            assert!((reconstructed - tensor.data()[col + 2 * row]).abs() < 1e-10);
        }
    }
}

fn f64_svd_outputs(rows: usize, cols: usize) -> Vec<DenseTensor> {
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let data = vec![1.0; rows * cols];
    dense
        .svd(DenseRead::F64(
            tenet_dense::DenseView::new(&data, &[rows, cols], &[1, rows], 0).unwrap(),
        ))
        .unwrap()
}

fn c64_svd_outputs(rows: usize, cols: usize) -> Vec<DenseTensor> {
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let data = vec![Complex64::new(1.0, 1.0); rows * cols];
    dense
        .svd(DenseRead::C64(
            tenet_dense::DenseView::new(&data, &[rows, cols], &[1, rows], 0).unwrap(),
        ))
        .unwrap()
}

#[test]
fn compact_owned_svd_preserves_svd_into_output_precedence() {
    let tensor = rectangular_svd_tensor(2, 2);
    let bound = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let input = bound.as_ref();
    let check = |outputs: Vec<DenseTensor>, expected: &str| {
        let mut dense = FailAfterObservingSvdInput {
            outputs: Some(outputs),
            ..Default::default()
        };
        let error = svd_compact(&mut dense, &input).unwrap_err();
        assert!(format!("{error}").contains(expected), "{error:?}");
    };

    check(vec![f64_svd_outputs(2, 2).remove(0)], "exactly (U, S, Vt)");
    check(
        f64_svd_outputs(1, 1),
        "output shape mismatch: source [1, 1], destination [2, 2]",
    );
    let mut outputs = f64_svd_outputs(2, 2);
    outputs[1] = f64_svd_outputs(1, 1).remove(1);
    check(
        outputs,
        "output shape mismatch: source [1], destination [2]",
    );
    let mut outputs = f64_svd_outputs(2, 2);
    outputs[2] = f64_svd_outputs(1, 1).remove(2);
    check(
        outputs,
        "output shape mismatch: source [1, 1], destination [2, 2]",
    );
    let outputs = c64_svd_outputs(2, 2);
    let expected = outputs[0].as_f64_slice().unwrap_err();
    let mut dense = FailAfterObservingSvdInput {
        outputs: Some(outputs),
        ..Default::default()
    };
    let error = svd_compact(&mut dense, &input).unwrap_err();
    assert!(matches!(error, OperationError::Dense(actual) if actual == expected));

    let mut outputs = f64_svd_outputs(2, 2);
    outputs[1] = c64_svd_outputs(2, 2).remove(0);
    let expected = outputs[1].as_f64_slice().unwrap_err();
    let mut dense = FailAfterObservingSvdInput {
        outputs: Some(outputs),
        ..Default::default()
    };
    let error = svd_compact(&mut dense, &input).unwrap_err();
    assert!(matches!(error, OperationError::Dense(actual) if actual == expected));

    let mut outputs = f64_svd_outputs(2, 2);
    outputs[2] = c64_svd_outputs(2, 2).remove(0);
    let expected = outputs[2].as_f64_slice().unwrap_err();
    let mut dense = FailAfterObservingSvdInput {
        outputs: Some(outputs),
        ..Default::default()
    };
    let error = svd_compact(&mut dense, &input).unwrap_err();
    assert!(matches!(error, OperationError::Dense(actual) if actual == expected));
}

#[test]
fn compact_svd_error_preserves_borrowed_input_and_publishes_no_factors() {
    // What: a provider failure cannot mutate borrowed tensor storage or return partial factors.
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let before = tensor.data().to_vec();
    let mut dense = FailAfterObservingSvdInput::default();

    let result = svd_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor));

    assert!(matches!(result, Err(OperationError::Dense(_))));
    assert_eq!(tensor.data(), before);
    assert!(!dense.observed.is_empty());
    assert!(dense
        .observed
        .iter()
        .all(|sector| before.windows(sector.len()).any(|window| window == sector)));
}

fn assert_rectangular_direct_svd(rows: usize, cols: usize) {
    let rule = Z2FusionRule;
    let tensor = rectangular_svd_tensor(rows, cols);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let bound = bound_tensor(Arc::new(rule), &tensor);
    crate::factorize::reset_compact_svd_copy_probe();
    let svd = svd_compact(&mut dense, &bound.as_ref()).unwrap();
    assert_factor_layout_matches_legacy_shapes(svd.u.space());
    assert_factor_layout_matches_legacy_shapes(svd.s.space());
    assert_factor_layout_matches_legacy_shapes(svd.vh.space());
    assert_compact_svd_direct_copy_probe();
    let rank = rows.min(cols);
    if rank == 0 {
        assert!(svd.u.space().space().homspace().domain().legs()[0]
            .sectors()
            .is_empty());
        assert!(svd.vh.space().space().homspace().codomain().legs()[0]
            .sectors()
            .is_empty());
    }
    let singular = svd
        .singular_values
        .first()
        .map(|entry| entry.values.as_slice())
        .unwrap_or_default();
    assert_eq!(singular.len(), rank);
    for col in 0..cols {
        for row in 0..rows {
            let reconstructed = (0..rank)
                .map(|bond| {
                    svd.u.data()[row + rows * bond]
                        * singular[bond]
                        * svd.vh.data()[bond + rank * col]
                })
                .sum::<f64>();
            assert!((reconstructed - tensor.data()[row + rows * col]).abs() < 1e-10);
        }
    }

    crate::factorize::reset_compact_svd_copy_probe();
    let adjoint = svd_compact_adjoint_factors_dyn(&mut dense, &bound.as_ref().dynamic()).unwrap();
    let adjoint_probe = crate::factorize::compact_svd_copy_probe();
    assert_eq!(adjoint_probe.input_pack_calls, 0);
    assert_eq!(adjoint_probe.output_scatter_calls, 0);
    if rank != 0 {
        assert_eq!(adjoint_probe.owned_output_owner_reused, 2);
    }
    let adjoint_singular = adjoint
        .2
        .first()
        .map(|entry| entry.values.as_slice())
        .unwrap_or_default();
    assert_eq!(adjoint_singular.len(), rank);
    for col in 0..rows {
        for row in 0..cols {
            let reconstructed = (0..rank)
                .map(|bond| {
                    adjoint.0.data()[row + cols * bond]
                        * adjoint_singular[bond]
                        * adjoint.1.data()[bond + rank * col]
                })
                .sum::<f64>();
            assert!((reconstructed - tensor.data()[col + rows * row]).abs() < 1e-10);
        }
    }
}

#[test]
fn compact_svd_direct_spans_reconstruct_tall_and_wide_matrices() {
    // What: exact final-factor spans work for both compact rectangular shapes.
    assert_rectangular_direct_svd(5, 3);
    assert_rectangular_direct_svd(3, 5);
}

#[test]
fn compact_svd_direct_outputs_keep_executor_factor_owners() {
    fn check<D: crate::factorize::FactorScalar>(tensor: &TensorMap<D, 1, 1>) {
        let mut dense = RejectSvdInto::default();
        let bound = bound_tensor(Arc::new(Z2FusionRule), tensor);
        crate::factorize::reset_compact_svd_copy_probe();
        let svd = svd_compact(&mut dense, &bound.as_ref()).unwrap();
        assert_eq!(dense.output_ptrs.len(), 1);
        let (u, vh) = dense.output_ptrs[0];
        assert_eq!(u, svd.u.data().as_ptr() as usize);
        assert_eq!(vh, svd.vh.data().as_ptr() as usize);
        let probe = crate::factorize::compact_svd_copy_probe();
        assert_compact_svd_direct_copy_probe();
        assert_eq!(probe.owned_output_publications, 2);
        assert_eq!(probe.owned_output_owner_reused, 2);
    }

    let tensor = rectangular_svd_tensor(3, 2);
    check(&tensor);
    let space = tensor.fusion_space().unwrap().as_ref().clone();
    check(
        &TensorMap::<f32, 1, 1>::from_vec_with_fusion_space(
            tensor.data().iter().map(|&value| value as f32).collect(),
            space.clone(),
        )
        .unwrap(),
    );
    check(
        &TensorMap::<Complex32, 1, 1>::from_vec_with_fusion_space(
            tensor
                .data()
                .iter()
                .map(|&value| Complex32::new(value as f32, value as f32 * 0.25))
                .collect(),
            space.clone(),
        )
        .unwrap(),
    );
    check(
        &TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
            tensor
                .data()
                .iter()
                .map(|&value| Complex64::new(value, value * 0.25))
                .collect(),
            space,
        )
        .unwrap(),
    );
}

#[test]
fn compact_svd_zero_only_input_normalizes_to_an_empty_factorization_result() {
    // What: a zero-only row or column produces empty factors and no phantom
    // spectrum entry or factor route.
    assert_rectangular_direct_svd(0, 3);
    assert_rectangular_direct_svd(3, 0);
}

#[test]
fn compact_svd_direct_and_fallback_apply_the_same_gauge() {
    // What: direct writes do not change the canonical phase chosen by the fallback.
    let rule = Z2FusionRule;
    let tensor = rectangular_svd_tensor(3, 3);
    let mut transposed_data = vec![0.0; 9];
    for col in 0..3 {
        for row in 0..3 {
            transposed_data[row + 3 * col] = tensor.data()[col + 3 * row];
        }
    }
    let transposed = TensorMap::from_vec_with_fusion_space(
        transposed_data,
        tensor.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let direct = svd_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &transposed)).unwrap();
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let adjoint_space = bound.space().adjoint_view().unwrap();
    let fallback_input = BoundDynamicTensorRef::try_new(&adjoint_space, bound.data()).unwrap();
    let fallback = svd_compact_dyn(&mut dense, &fallback_input).unwrap();

    for (left, right) in direct.u.data().iter().zip(fallback.u().data()) {
        assert!((left - right).abs() < 1e-12);
    }
    for (left, right) in direct.vh.data().iter().zip(fallback.vh().data()) {
        assert!((left - right).abs() < 1e-12);
    }
    assert_spectra_agree(
        "direct vs adjoint-view fallback",
        &direct.singular_values,
        fallback.singular_values(),
    );

    let adjoint_fallback = svd_compact_adjoint_factors_dyn(&mut dense, &fallback_input).unwrap();
    let expected = svd_compact_factors_dyn(&mut dense, &bound.as_ref().dynamic()).unwrap();
    for (actual, expected) in adjoint_fallback.0.data().iter().zip(expected.0.data()) {
        assert!((actual - expected).abs() < 1e-12);
    }
    for (actual, expected) in adjoint_fallback.1.data().iter().zip(expected.1.data()) {
        assert!((actual - expected).abs() < 1e-12);
    }
    for (actual, expected) in adjoint_fallback.2.iter().zip(&expected.2) {
        assert_eq!(actual.sector, expected.sector);
        for (actual, expected) in actual.values.iter().zip(&expected.values) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
}

#[test]
fn compact_svd_adjoint_accepts_padded_parent_layout() {
    // What: the optimized adjoint path uses the existing packed fallback for
    // custom offsets instead of materializing a canonical adjoint input.
    let rule = Z2FusionRule;
    let parent = padded_copy(&rule, &rectangular_svd_tensor(5, 3));
    let bound = bound_tensor(Arc::new(rule), &parent);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let actual = svd_compact_adjoint_factors_dyn(&mut dense, &bound.as_ref().dynamic()).unwrap();
    let singular = &actual.2[0].values;
    let source = parent.structure().block(0).unwrap();
    for col in 0..5 {
        for row in 0..3 {
            let reconstructed = (0..3)
                .map(|bond| {
                    actual.0.data()[row + 3 * bond]
                        * singular[bond]
                        * actual.1.data()[bond + 3 * col]
                })
                .sum::<f64>();
            let expected = parent.data()
                [source.offset() + col * source.strides()[0] + row * source.strides()[1]];
            assert!((reconstructed - expected).abs() < 1e-10);
        }
    }
}

#[test]
fn compact_svd_adjoint_c64_padded_parent_reconstructs_literal_adjoint() {
    let rule = Z2FusionRule;
    let source = rectangular_svd_tensor(5, 3);
    let complex = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        source
            .data()
            .iter()
            .enumerate()
            .map(|(index, &value)| Complex64::new(value, index as f64 * 0.125 - 0.5))
            .collect(),
        source.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let parent = padded_copy(&rule, &complex);
    let bound = bound_tensor(Arc::new(rule), &parent);
    assert!(
        crate::factorize::compact_factor_plan_for_test(bound.space())
            .unwrap()
            .is_none()
    );
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    crate::factorize::reset_compact_svd_copy_probe();
    let actual = svd_compact_adjoint_factors_dyn(&mut dense, &bound.as_ref().dynamic()).unwrap();
    let singular = &actual.2[0].values;
    let source = parent.structure().block(0).unwrap();
    for col in 0..5 {
        for row in 0..3 {
            let reconstructed = (0..3)
                .map(|bond| {
                    actual.0.data()[row + 3 * bond]
                        * singular[bond]
                        * actual.1.data()[bond + 3 * col]
                })
                .sum::<Complex64>();
            let expected = parent.data()
                [source.offset() + col * source.strides()[0] + row * source.strides()[1]]
                .conj();
            assert!((reconstructed - expected).norm() < 1e-10);
        }
    }
    let probe = crate::factorize::compact_svd_copy_probe();
    assert!(probe.input_pack_calls > 0);
    assert!(probe.output_scatter_calls > 0);
}

#[test]
fn compact_svd_c64_reconstructs_mixed_tall_and_wide_sectors_without_copies() {
    use num_complex::Complex64;

    // What: one call reconstructs mixed rectangular complex sectors directly in final storage.
    let rule = Z2FusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(even, 5), (odd, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(even, 3), (odd, 4)], false)]),
    );
    let shapes = homspace
        .fusion_tree_keys(&rule)
        .iter()
        .map(|key| match key.codomain_tree().coupled() {
            sector if sector == even => vec![5, 3],
            sector if sector == odd => vec![2, 4],
            sector => panic!("unexpected Z2 sector {sector:?}"),
        })
        .collect::<Vec<_>>();
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([7], [7]).unwrap(),
        homspace,
        &rule,
        shapes,
    )
    .unwrap();
    let tensor = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        (0..space.required_len().unwrap())
            .map(|index| {
                Complex64::new(
                    ((index * 7 + 2) % 17) as f64 - 6.0,
                    ((index * 5 + 3) % 13) as f64 * 0.25 - 1.0,
                )
            })
            .collect(),
        space,
    )
    .unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    crate::factorize::reset_compact_svd_copy_probe();

    let svd = svd_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();

    assert_compact_svd_direct_copy_probe();
    let input_regions = tensor
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let u_regions = svd
        .u
        .tensor()
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let vh_regions = svd
        .vh
        .tensor()
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    for input_region in input_regions.iter() {
        let sector = input_region.coupled();
        let u_region = u_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let vh_region = vh_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let singular = &svd
            .singular_values
            .iter()
            .find(|values| values.sector == sector)
            .unwrap()
            .values;
        let rows = input_region.rows();
        let cols = input_region.cols();
        let rank = rows.min(cols);
        for col in 0..cols {
            for row in 0..rows {
                let reconstructed = (0..rank)
                    .map(|bond| {
                        svd.u.data()[u_region.range().start + row + rows * bond]
                            * singular[bond]
                            * svd.vh.data()[vh_region.range().start + bond + rank * col]
                    })
                    .sum::<Complex64>();
                let expected = tensor.data()[input_region.range().start + row + rows * col];
                assert!((reconstructed - expected).norm() < 1e-10);
            }
        }
    }
}

#[test]
fn compact_svd_adjoint_c64_padded_fallback_matches_canonical_gauge() {
    let rule = Z2FusionRule;
    let source = mixed_rectangular_c32_tensor();
    let canonical = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        source
            .data()
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect(),
        source.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let padded = padded_copy(&rule, &canonical);
    let canonical_bound = bound_tensor(Arc::new(rule), &canonical);
    let padded_bound = bound_tensor(Arc::new(rule), &padded);
    assert!(
        crate::factorize::compact_factor_plan_for_test(canonical_bound.space())
            .unwrap()
            .is_some()
    );
    assert!(
        crate::factorize::compact_factor_plan_for_test(padded_bound.space())
            .unwrap()
            .is_none()
    );
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let expected =
        svd_compact_adjoint_factors_dyn(&mut dense, &canonical_bound.as_ref().dynamic()).unwrap();
    crate::factorize::reset_compact_svd_copy_probe();
    let actual =
        svd_compact_adjoint_factors_dyn(&mut dense, &padded_bound.as_ref().dynamic()).unwrap();

    assert_eq!(
        actual.0.space().space().structure(),
        expected.0.space().space().structure()
    );
    assert_eq!(
        actual.1.space().space().structure(),
        expected.1.space().space().structure()
    );
    for (&actual, &expected) in actual.0.data().iter().zip(expected.0.data()) {
        assert!((actual - expected).norm() < 1.0e-10);
    }
    for (&actual, &expected) in actual.1.data().iter().zip(expected.1.data()) {
        assert!((actual - expected).norm() < 1.0e-10);
    }
    assert_real_spectra_close(&actual.2, &expected.2);
    let probe = crate::factorize::compact_svd_copy_probe();
    assert!(probe.input_pack_calls > 0);
    assert!(probe.output_scatter_calls > 0);
}

#[test]
fn compact_svd_c32_reconstructs_mixed_tall_and_wide_sectors_without_copies() {
    // What: single-precision complex direct spans reconstruct both rectangular orientations.
    let rule = Z2FusionRule;
    let tensor = mixed_rectangular_c32_tensor();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    crate::factorize::reset_compact_svd_copy_probe();

    let svd = svd_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();

    assert_compact_svd_direct_copy_probe();
    let input_regions = tensor
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let u_regions = svd
        .u
        .tensor()
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let vh_regions = svd
        .vh
        .tensor()
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    for input_region in input_regions.iter() {
        let sector = input_region.coupled();
        let u_region = u_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let vh_region = vh_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let singular = &svd
            .singular_values
            .iter()
            .find(|values| values.sector == sector)
            .unwrap()
            .values;
        let rows = input_region.rows();
        let cols = input_region.cols();
        let rank = rows.min(cols);
        for col in 0..cols {
            for row in 0..rows {
                let reconstructed = (0..rank)
                    .map(|bond| {
                        svd.u.data()[u_region.range().start + row + rows * bond]
                            * singular[bond] as f32
                            * svd.vh.data()[vh_region.range().start + bond + rank * col]
                    })
                    .sum::<Complex32>();
                let expected = tensor.data()[input_region.range().start + row + rows * col];
                assert!((reconstructed - expected).norm() < 2e-4);
            }
        }
    }
}

#[test]
fn compact_svd_c32_direct_and_fallback_apply_the_same_gauge() {
    // What: the single-precision direct path preserves the fallback's canonical complex phase.
    let rule = Z2FusionRule;
    let real = rectangular_svd_tensor(3, 3);
    let data = real
        .data()
        .iter()
        .enumerate()
        .map(|(index, &value)| Complex32::new(value as f32, (index as f32 - 3.0) * 0.25))
        .collect::<Vec<_>>();
    let tensor =
        TensorMap::from_vec_with_fusion_space(data, real.fusion_space().unwrap().as_ref().clone())
            .unwrap();
    let mut transposed_data = vec![Complex32::new(0.0, 0.0); 9];
    for col in 0..3 {
        for row in 0..3 {
            transposed_data[row + 3 * col] = tensor.data()[col + 3 * row];
        }
    }
    let transposed = TensorMap::from_vec_with_fusion_space(
        transposed_data,
        tensor.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    crate::factorize::reset_compact_svd_copy_probe();
    let direct = svd_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &transposed)).unwrap();
    assert_compact_svd_direct_copy_probe();
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let adjoint_space = bound.space().adjoint_view().unwrap();
    let fallback_input = BoundDynamicTensorRef::try_new(&adjoint_space, bound.data()).unwrap();
    crate::factorize::reset_compact_svd_copy_probe();
    let fallback = svd_compact_dyn(&mut dense, &fallback_input).unwrap();
    let fallback_probe = crate::factorize::compact_svd_copy_probe();
    assert!(fallback_probe.input_pack_calls > 0);
    assert!(fallback_probe.output_scatter_calls > 0);
    for entry in &direct.singular_values {
        assert!(entry.values.last().is_some_and(|value| *value > 1e-3));
        assert!(entry
            .values
            .windows(2)
            .all(|pair| (pair[0] - pair[1]).abs() > 1e-3));
    }

    assert_eq!(direct.u.data().len(), fallback.u().data().len());
    for (left, right) in direct.u.data().iter().zip(fallback.u().data()) {
        assert!((*left - *right).norm() < 2e-5);
    }
    assert_eq!(direct.vh.data().len(), fallback.vh().data().len());
    for (left, right) in direct.vh.data().iter().zip(fallback.vh().data()) {
        assert!((*left - *right).norm() < 2e-5);
    }
    assert_eq!(
        direct.singular_values.len(),
        fallback.singular_values().len()
    );
    for (left_entry, right_entry) in direct
        .singular_values
        .iter()
        .zip(fallback.singular_values())
    {
        assert_eq!(left_entry.sector, right_entry.sector);
        assert_eq!(left_entry.values.len(), right_entry.values.len());
        for (left, right) in left_entry.values.iter().zip(&right_entry.values) {
            assert!((left - right).abs() < 1e-5);
        }
    }
}

#[test]
fn svd_compact_factor_dims_include_sectors_without_populated_trees() {
    // What: public compact SVD retains the complete original leg space,
    // including a sector absent from every populated fusion block.
    let rule = U1FusionRule;
    let neutral = U1Irrep::new(0).sector_id();
    let positive = U1Irrep::new(1).sector_id();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(neutral, 2), (positive, 3)], false)]),
        FusionProductSpace::new([SectorLeg::new([(neutral, 2)], false)]),
    );
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([5], [2]).unwrap(),
        homspace,
        &rule,
        [vec![2, 2]],
    )
    .unwrap();
    let tensor = TensorMap::from_vec_with_fusion_space(vec![1.0, 2.0, 3.0, 4.0], space).unwrap();
    let original_homspace = tensor.fusion_space().unwrap().homspace().clone();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let result = svd_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();

    assert_factor_layout_matches_legacy_shapes(result.u.space());
    assert_factor_layout_matches_legacy_shapes(result.s.space());
    assert_factor_layout_matches_legacy_shapes(result.vh.space());
    assert_eq!(result.u.tensor().space().dims(), &[5, 2]);
    assert_eq!(result.vh.tensor().space().dims(), &[2, 2]);
    assert_eq!(
        result
            .u
            .tensor()
            .fusion_space()
            .unwrap()
            .homspace()
            .codomain(),
        original_homspace.codomain()
    );
    assert_eq!(
        result
            .vh
            .tensor()
            .fusion_space()
            .unwrap()
            .homspace()
            .domain(),
        original_homspace.domain()
    );
}

#[test]
fn svd_compact_preserves_asymmetric_non_self_dual_u1_factor_layouts() {
    // What: canonical factors retain unequal degeneracies and the dual U(1)
    // domain convention while reconstructing both coupled sectors.
    let rule = U1FusionRule;
    let neutral = U1Irrep::new(0).sector_id();
    let positive = U1Irrep::new(1).sector_id();
    let negative = U1Irrep::new(-1).sector_id();
    let codomain = SectorLeg::new([(neutral, 3), (positive, 2)], false);
    let domain = SectorLeg::new([(neutral, 1), (negative, 4)], true);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([codomain]),
        FusionProductSpace::new([domain]),
    );
    let shapes = homspace
        .fusion_tree_keys(&rule)
        .iter()
        .map(|key| match key.codomain_tree().coupled() {
            sector if sector == neutral => vec![3, 1],
            sector if sector == positive => vec![2, 4],
            sector => panic!("unexpected U(1) sector {sector:?}"),
        })
        .collect::<Vec<_>>();
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([5], [5]).unwrap(),
        homspace,
        &rule,
        shapes,
    )
    .unwrap();
    let tensor = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        (0..space.required_len().unwrap())
            .map(|index| ((index * 7 + 3) % 17) as f64 - 6.0)
            .collect(),
        space,
    )
    .unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let svd = svd_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();

    assert_factor_layout_matches_legacy_shapes(svd.u.space());
    assert_factor_layout_matches_legacy_shapes(svd.s.space());
    assert_factor_layout_matches_legacy_shapes(svd.vh.space());
    let input_regions = tensor
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let u_regions = svd
        .u
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let vh_regions = svd
        .vh
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    for input_region in input_regions.iter() {
        let sector = input_region.coupled();
        let u_region = u_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let vh_region = vh_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let singular = &svd
            .singular_values
            .iter()
            .find(|entry| entry.sector == sector)
            .unwrap()
            .values;
        for col in 0..input_region.cols() {
            for row in 0..input_region.rows() {
                let reconstructed = (0..singular.len())
                    .map(|bond| {
                        svd.u.data()[u_region.range().start + row + input_region.rows() * bond]
                            * singular[bond]
                            * svd.vh.data()[vh_region.range().start + bond + singular.len() * col]
                    })
                    .sum::<f64>();
                let expected =
                    tensor.data()[input_region.range().start + row + input_region.rows() * col];
                assert!((reconstructed - expected).abs() < 1.0e-10);
            }
        }
    }
}

#[test]
fn svd_rejects_a_different_provider_before_dense_execution() {
    let tensor = hermitian_test_tensor(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);

    let _backend = RejectExecutorCalls;
    let error = match BoundTensorMap::try_new(Arc::new(U1FusionRule), tensor) {
        Ok(_) => panic!("mismatched provider must not produce an authority"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionRuleMismatch { .. })
    ));
}

#[test]
fn svd_full_rejects_a_different_provider_before_dense_execution() {
    let tensor = tsvd_test_tensor(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);

    let _backend = RejectExecutorCalls;
    let error = match BoundTensorMap::try_new(Arc::new(U1FusionRule), tensor) {
        Ok(_) => panic!("mismatched provider must not produce an authority"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionRuleMismatch { .. })
    ));
}

#[test]
fn public_svd_authority_rejects_same_type_with_different_identity_and_qdim() {
    // What: provider provenance, not the Rust type or sector ids, owns qdim.
    let source_rule = IdentityQdimRule::new(1.0);
    let other_rule = IdentityQdimRule::new((1.0 + 5.0_f64.sqrt()) / 2.0);
    let tensor = tsvd_test_tensor(&source_rule, &[SectorId::new(0)]);

    let _backend = RejectExecutorCalls;
    let error = match BoundTensorMap::try_new(Arc::new(other_rule), tensor) {
        Ok(_) => panic!("different provider identity must not produce an authority"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionRuleMismatch { .. })
    ));
}

#[test]
fn svd_input_rejects_short_storage_before_dense_execution() {
    let tensor = tsvd_test_tensor(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        dyn_space_of(&tensor).unwrap(),
        Arc::new(Z2FusionRule),
    )
    .unwrap();
    let short = &tensor.data()[..tensor.data().len() - 1];

    let error = match BoundDynamicTensorRef::try_new(&bound, short) {
        Ok(_) => panic!("short storage must be rejected"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        OperationError::Core(CoreError::DimensionMismatch { .. })
    ));
}

#[test]
fn typed_svd_borrows_input_authority_and_retains_its_exact_allocation() {
    // What: borrowed typed input creates no replacement authority, and every SVD factor inherits it.
    let rule = Z2FusionRule;
    let provider = Arc::new(rule);
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let input = bound_tensor(Arc::clone(&provider), &tensor);
    let first = input.as_ref();
    let second = input.as_ref();

    assert!(std::ptr::eq(first.space(), input.space()));
    assert!(std::ptr::eq(second.space(), input.space()));
    assert!(std::ptr::eq(first.tensor(), input.tensor()));
    assert!(std::ptr::eq(second.tensor(), input.tensor()));

    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let factors = svd_compact(&mut dense, &first).unwrap();
    assert!(Arc::ptr_eq(&provider, factors.u.space().provider_arc()));
    assert!(Arc::ptr_eq(&provider, factors.s.space().provider_arc()));
    assert!(Arc::ptr_eq(&provider, factors.vh.space().provider_arc()));
}

#[test]
fn svd_compact_dense_failure_preserves_input_and_builds_no_diagonal_factor() {
    // What: a failed dense SVD leaves borrowed input unchanged and cannot publish or build factors.
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let before = tensor.data().to_vec();
    let mut dense = FailAfterObservingSvdInput::default();

    crate::factorize::reset_diagonal_bond_build_probe();
    let result = svd_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor));

    assert!(matches!(result, Err(OperationError::Dense(_))));
    assert_eq!(tensor.data(), before);
    assert_eq!(
        crate::factorize::diagonal_bond_build_probe(),
        crate::factorize::DiagonalBondBuildProbe::default()
    );
}

fn assert_zero_axis_svd_compact(rows: usize, cols: usize) {
    let rule = Z2FusionRule;
    let tensor = rectangular_svd_tensor(rows, cols);
    let input = bound_tensor(Arc::new(rule), &tensor);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    {
        crate::factorize::reset_diagonal_bond_build_probe();
        let result = svd_compact_dyn(&mut dense, &input.as_ref().dynamic()).unwrap();
        assert_eq!(
            result
                .singular_values()
                .iter()
                .map(|entry| entry.values.len())
                .sum::<usize>(),
            0
        );
        assert!(result.u().data().is_empty());
        assert!(result.s().data().is_empty());
        assert!(result.vh().data().is_empty());
        for factor in [result.u(), result.s(), result.vh()] {
            assert_eq!(factor.space().space().structure().block_count(), 0);
            assert_factor_layout_matches_legacy_shapes(factor.space());
        }
        assert_eq!(
            crate::factorize::diagonal_bond_build_probe(),
            crate::factorize::DiagonalBondBuildProbe {
                calls: 1,
                values: 0,
            }
        );
    }
}

#[test]
fn svd_compact_zero_only_input_normalizes_to_an_empty_factorization_result() {
    // What: the compact SVD exposes no phantom sector when either side of the
    // zero-only input is absent.
    assert_zero_axis_svd_compact(0, 3);
    assert_zero_axis_svd_compact(3, 0);
}

#[test]
fn svd_full_gives_square_unitaries_and_reconstructs() {
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let full = svd_full(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();

    let matrices = dense_sector_matrices(2, &full.u);
    for (_, rows, cols, _) in &matrices {
        assert_eq!(rows, cols, "full U must be square per sector");
    }
    assert_orthonormal_columns(&matrices);

    // U . S has U's codomain and S's (column) bond as domain; build its space
    // from the contraction homspace and per-tree shapes.
    let us_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        full.u.fusion_space().unwrap().homspace(),
        full.s.fusion_space().unwrap().homspace(),
        &[2],
        &[0],
        &[0, 1, 2],
        2,
    )
    .unwrap();
    let u_structure = std::sync::Arc::clone(full.u.structure());
    let s_structure = std::sync::Arc::clone(full.s.structure());
    let shapes = us_hom
        .fusion_tree_keys(&rule)
        .iter()
        .map(|key| {
            let sector = key.domain_tree().coupled();
            let mut shape = None;
            for index in 0..u_structure.block_count() {
                let block = u_structure.block(index).unwrap();
                let BlockKey::FusionTree(u_key) = block.key() else {
                    continue;
                };
                if u_key.codomain_tree() == key.codomain_tree() {
                    shape = Some(block.shape()[..2].to_vec());
                    break;
                }
            }
            let mut shape = shape.expect("U tree present");
            let mut s_cols = 0;
            for index in 0..s_structure.block_count() {
                let block = s_structure.block(index).unwrap();
                let BlockKey::FusionTree(s_key) = block.key() else {
                    continue;
                };
                let s_sector = s_key.domain_tree().coupled();
                if s_sector == sector {
                    s_cols = block.shape()[1];
                    break;
                }
            }
            shape.push(s_cols);
            shape
        })
        .collect::<Vec<_>>();
    let dims = full.u.tensor().space().dims();
    let us_space = FusionTensorMapSpace::<2, 1>::from_degeneracy_shapes_coupled(
        TensorMapSpace::<2, 1>::from_dims([dims[0], dims[1]], [full.s.tensor().space().dims()[1]])
            .unwrap(),
        us_hom,
        &rule,
        shapes,
    )
    .unwrap();
    let mut us = TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(
        vec![0.0; us_space.required_len().unwrap()],
        us_space,
    )
    .unwrap();
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    context
        .tensorcontract_fusion_into(
            &rule,
            &mut us,
            &full.u,
            &full.s,
            TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2])),
            1.0,
            0.0,
        )
        .unwrap();
    let reconstructed = contract_pair(&rule, &tensor, &us, &full.vh);
    assert_svd_blocks_match(&tensor, &reconstructed);
}

#[test]
fn svd_full_uses_native_owned_full_svd_without_legacy_completion() {
    let tensor = rectangular_svd_tensor(2, 3);
    let input = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let Some(mut dense) = NativeFullSvdSpy::new() else {
        return;
    };

    let full = svd_full_dyn(&mut dense, &input.as_ref().dynamic()).unwrap();

    assert_eq!(dense.full_calls, 1);
    assert!(!full.singular_values().is_empty());
}

fn assert_native_full_svd_uses_builtin_owned_dtype<D: FactorScalar>() {
    let source = mixed_rectangular_c32_tensor();
    let tensor = TensorMap::<D, 1, 1>::from_vec_with_fusion_space(
        source
            .data()
            .iter()
            .map(|value| D::from_complex64(Complex64::new(value.re as f64, value.im as f64)))
            .collect(),
        source.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let input = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let Some(mut dense) = NativeFullSvdSpy::new() else {
        return;
    };

    let full = svd_full_dyn(&mut dense, &input.as_ref().dynamic()).unwrap();

    assert_eq!(dense.full_calls, 2);
    assert_eq!(full.singular_values().len(), 2);
    assert!(full
        .singular_values()
        .iter()
        .all(|entry| !entry.values.is_empty()));
}

#[test]
fn native_full_svd_uses_owned_inputs_for_every_builtin_dtype() {
    assert_native_full_svd_uses_builtin_owned_dtype::<f32>();
    assert_native_full_svd_uses_builtin_owned_dtype::<f64>();
    assert_native_full_svd_uses_builtin_owned_dtype::<Complex32>();
    assert_native_full_svd_uses_builtin_owned_dtype::<Complex64>();
}

#[test]
fn native_full_svd_reconstructs_complex_mixed_rectangular_sectors() {
    let source = mixed_rectangular_c32_tensor();
    let tensor = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        source
            .data()
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect(),
        source.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let input = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let Some(mut dense) = NativeFullSvdSpy::new() else {
        return;
    };

    let full = svd_full_dyn(&mut dense, &input.as_ref().dynamic()).unwrap();
    assert_eq!(dense.full_calls, 2);

    let input_regions = tensor
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let u_regions = full
        .u()
        .space()
        .space()
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let s_regions = full
        .s()
        .space()
        .space()
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let vh_regions = full
        .vh()
        .space()
        .space()
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    for input_region in input_regions.iter() {
        let sector = input_region.coupled();
        let u_region = u_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let s_region = s_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let vh_region = vh_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let rows = input_region.rows();
        let cols = input_region.cols();
        assert_eq!(u_region.rows(), rows);
        assert_eq!(u_region.cols(), rows);
        assert_eq!(s_region.rows(), rows);
        assert_eq!(s_region.cols(), cols);
        assert_eq!(vh_region.rows(), cols);
        assert_eq!(vh_region.cols(), cols);

        for column in 0..rows {
            for row in 0..rows {
                let gram = (0..rows)
                    .map(|inner| {
                        full.u().data()[u_region.range().start + inner + rows * row].conj()
                            * full.u().data()[u_region.range().start + inner + rows * column]
                    })
                    .sum::<Complex64>();
                let expected = if row == column {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::zero()
                };
                assert!((gram - expected).norm() < 1.0e-10);
            }
        }
        for column in 0..cols {
            for row in 0..cols {
                let gram = (0..cols)
                    .map(|inner| {
                        full.vh().data()[vh_region.range().start + row + cols * inner]
                            * full.vh().data()[vh_region.range().start + column + cols * inner]
                                .conj()
                    })
                    .sum::<Complex64>();
                let expected = if row == column {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::zero()
                };
                assert!((gram - expected).norm() < 1.0e-10);
            }
        }

        let mut us = vec![Complex64::zero(); rows * cols];
        for col in 0..cols {
            for inner in 0..rows {
                for row in 0..rows {
                    us[row + rows * col] += full.u().data()
                        [u_region.range().start + row + rows * inner]
                        * full.s().data()[s_region.range().start + inner + rows * col];
                }
            }
        }
        for col in 0..cols {
            for row in 0..rows {
                let actual = (0..cols)
                    .map(|inner| {
                        us[row + rows * inner]
                            * full.vh().data()[vh_region.range().start + inner + cols * col]
                    })
                    .sum::<Complex64>();
                let expected = tensor.data()[input_region.range().start + row + rows * col];
                assert!((actual - expected).norm() < 1.0e-10);
            }
        }
    }
}

/// The single coupled sector of a one-sector fixture, widened to `Complex64`
/// so one oracle serves every supported dtype.
fn single_sector_block<R, D>(factor: &BoundDynFactor<R, D>) -> (usize, usize, Vec<Complex64>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let space = factor.space().space();
    let regions = space
        .structure()
        .coupled_sector_regions(space.nout())
        .unwrap()
        .unwrap();
    let mut regions = regions.iter();
    let region = regions.next().expect("the oracle fixture has one sector");
    assert!(
        regions.next().is_none(),
        "the oracle fixture has one sector"
    );
    (
        region.rows(),
        region.cols(),
        factor.data()[region.range()]
            .iter()
            .copied()
            .map(FactorScalar::widen_complex)
            .collect(),
    )
}

/// Independent full-SVD oracle: reconstruction, unitarity of both square
/// factors, and the Frobenius identity for the spectrum. Never factor entries
/// — the basis spanning the null space is not unique, so it differs between
/// the direct provider call and the `[U1 | I]` completion fallback.
fn assert_native_full_svd_oracle<D: FactorScalar>(tol: f64) {
    for (rows, cols) in [(3usize, 3usize), (4, 2), (2, 4)] {
        for kind in ["generic", "rank-deficient", "repeated"] {
            let entry = |index: usize| {
                let (row, col) = (index % rows, index / rows);
                match kind {
                    // every column is a fixed multiple of the first: rank 1.
                    "rank-deficient" => {
                        let value = (((row * 7 + 3) % 13) as f64 - 6.0) * (col as f64 + 1.0);
                        Complex64::new(value, 0.5 * value)
                    }
                    // sigma repeated min(rows, cols) times.
                    "repeated" => {
                        let value = if row == col { 2.0 } else { 0.0 };
                        Complex64::new(value, 0.5 * value)
                    }
                    _ => Complex64::new(
                        ((index * 11 + 2) % 19) as f64 - 7.0,
                        ((index * 5 + 1) % 17) as f64 - 8.0,
                    ),
                }
            };
            let data = (0..rows * cols)
                .map(|index| D::from_complex64(entry(index)))
                .collect::<Vec<_>>();
            let expected = data
                .iter()
                .copied()
                .map(FactorScalar::widen_complex)
                .collect::<Vec<_>>();
            let norm = expected
                .iter()
                .map(|value| value.norm_sqr())
                .sum::<f64>()
                .sqrt();
            let tensor = one_sector_rectangular_matrix(data, rows, cols);
            let input = bound_tensor(Arc::new(Z2FusionRule), &tensor);
            let Some(mut dense) = NativeFullSvdSpy::new() else {
                return;
            };

            let full = svd_full_dyn(&mut dense, &input.as_ref().dynamic()).unwrap();

            let context = format!("{kind} {rows}x{cols}");
            assert_eq!(dense.full_calls, 1, "{context}: direct provider call");
            let (u_rows, u_cols, u) = single_sector_block(full.u());
            let (s_rows, s_cols, s) = single_sector_block(full.s());
            let (vh_rows, vh_cols, vh) = single_sector_block(full.vh());
            assert_eq!((u_rows, u_cols), (rows, rows), "{context}: U shape");
            assert_eq!((s_rows, s_cols), (rows, cols), "{context}: S shape");
            assert_eq!((vh_rows, vh_cols), (cols, cols), "{context}: Vh shape");

            for left in 0..rows {
                for right in 0..rows {
                    let gram = (0..rows)
                        .map(|inner| u[inner + rows * left].conj() * u[inner + rows * right])
                        .sum::<Complex64>();
                    let unit = f64::from(u8::from(left == right));
                    assert!(
                        (gram - Complex64::new(unit, 0.0)).norm() <= tol,
                        "{context}: U column dot ({left},{right}) = {gram}"
                    );
                }
            }
            for upper in 0..cols {
                for lower in 0..cols {
                    let gram = (0..cols)
                        .map(|inner| vh[upper + cols * inner] * vh[lower + cols * inner].conj())
                        .sum::<Complex64>();
                    let unit = f64::from(u8::from(upper == lower));
                    assert!(
                        (gram - Complex64::new(unit, 0.0)).norm() <= tol,
                        "{context}: Vh row dot ({upper},{lower}) = {gram}"
                    );
                }
            }

            let rank = rows.min(cols);
            for col in 0..cols {
                for row in 0..rows {
                    let actual = (0..rank)
                        .map(|inner| {
                            u[row + rows * inner] * s[inner + rows * inner] * vh[inner + cols * col]
                        })
                        .sum::<Complex64>();
                    assert!(
                        (actual - expected[row + rows * col]).norm() <= tol * norm,
                        "{context}: reconstruction ({row},{col})"
                    );
                }
            }

            let spectra = full.singular_values();
            assert_eq!(spectra.len(), 1, "{context}: one sector spectrum");
            let values = &spectra[0].values;
            assert_eq!(values.len(), rank, "{context}: spectrum length");
            for (index, value) in values.iter().enumerate() {
                assert!(*value >= 0.0, "{context}: sigma {index} is negative");
                assert!(
                    index == 0 || values[index - 1] >= *value,
                    "{context}: sigma {index} breaks descending order"
                );
                assert!(
                    (s[index + rows * index] - Complex64::new(*value, 0.0)).norm() <= tol * norm,
                    "{context}: S diagonal {index} disagrees with the spectrum"
                );
            }
            // Independent of the factors: the Frobenius norm is the 2-norm of
            // the spectrum, so no singular value may be missing or spurious.
            let spectrum_norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
            assert!(
                (spectrum_norm - norm).abs() <= tol * norm.max(1.0),
                "{context}: ||sigma||_2 = {spectrum_norm} but ||A||_F = {norm}"
            );
            match kind {
                "rank-deficient" => {
                    assert!(values[0] > tol, "{context}: the rank-1 sigma vanished");
                    assert!(
                        values[1..].iter().all(|value| *value <= tol * norm),
                        "{context}: a rank-deficient sigma is nonzero"
                    );
                }
                "repeated" => assert!(
                    (values[0] - values[rank - 1]).abs() <= tol * norm,
                    "{context}: repeated singular values disagree"
                ),
                _ => assert!(values[rank - 1] > tol, "{context}: unexpected rank loss"),
            }
        }
    }
}

#[test]
fn native_full_svd_oracle_holds_for_every_builtin_dtype_shape_and_rank() {
    assert_native_full_svd_oracle::<f32>(1.0e-4);
    assert_native_full_svd_oracle::<f64>(1.0e-10);
    assert_native_full_svd_oracle::<Complex32>(1.0e-4);
    assert_native_full_svd_oracle::<Complex64>(1.0e-10);
}

#[test]
fn native_full_svd_late_failure_does_not_publish_or_retry_compatibility() {
    let tensor = mixed_rectangular_c32_tensor();
    let before = tensor.data().to_vec();
    let input = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let Some(mut dense) = FailSecondOwnedFullSvd::new() else {
        return;
    };

    crate::factorize::reset_factor_buffer_build_counts_for_test();
    let result = svd_full_dyn(&mut dense, &input.as_ref().dynamic());

    assert!(matches!(result, Err(OperationError::Dense(_))));
    assert_eq!(dense.calls, 2);
    assert_eq!(input.data(), before);
    assert_eq!(
        crate::factorize::factor_buffer_build_counts_for_test(),
        (0, 0)
    );
}

#[test]
fn native_full_svd_reconstructs_complex_square_and_padded_inputs() {
    let square = one_sector_rectangular_matrix(
        vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(-3.0, 1.0),
            Complex64::new(0.5, -1.5),
            Complex64::new(2.0, 0.25),
        ],
        2,
        2,
    );
    for tensor in [&square, &padded_copy(&Z2FusionRule, &square)] {
        let input = bound_tensor(Arc::new(Z2FusionRule), tensor);
        let Some(mut dense) = NativeFullSvdSpy::new() else {
            return;
        };

        let full = svd_full_dyn(&mut dense, &input.as_ref().dynamic()).unwrap();

        assert_eq!(dense.full_calls, 1);
        assert_compact_factors_reconstruct_input(
            &input.as_ref().dynamic(),
            full.u(),
            Some(full.s()),
            full.vh(),
        );
    }
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_native_full_svd_reconstructs_complex_interleaved_square_trees() {
    let (canonical, _, data) = generic_values_endomorphism_input();
    let (interleaved_space, interleaved_data) =
        interleaved_generic_endomorphism_input(&canonical, &data);
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let space = BoundDynamicFusionMapSpace::bind_generic(
        interleaved_space.space().clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let input = BoundDynamicTensorRef::try_new(&space, &interleaved_data).unwrap();
    let Some(mut dense) = NativeFullSvdSpy::new() else {
        return;
    };

    let full = svd_full_dyn_checked_generic(&mut dense, &input).unwrap();

    assert_eq!(dense.full_calls, 2);
    assert_compact_factors_reconstruct_input(&input, full.u(), Some(full.s()), full.vh());
    assert!(Arc::ptr_eq(full.u().space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(full.s().space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(full.vh().space().provider_arc(), &provider));
}

#[test]
fn rectangular_full_svd_has_square_outer_factors_and_reconstructs() {
    // What: full SVD returns U(m,m), S(m,n), Vh(n,n) and recomposes tall and wide inputs.
    let rule = Z2FusionRule;
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    for (rows, cols) in [(2, 3), (3, 2)] {
        let matrix = one_sector_rectangular_matrix(
            (0..rows * cols)
                .map(|index| ((index * 5 + 1) % 11) as f64 - 4.0)
                .collect(),
            rows,
            cols,
        );
        let input = bound_tensor(Arc::new(rule), &matrix);
        let full = svd_full(&mut dense, &input.as_ref()).unwrap();
        assert_factor_layout_matches_legacy_shapes(full.u.space());
        assert_factor_layout_matches_legacy_shapes(full.s.space());
        assert_factor_layout_matches_legacy_shapes(full.vh.space());
        assert_eq!(full.u.structure().block(0).unwrap().shape(), &[rows, rows]);
        assert_eq!(full.s.structure().block(0).unwrap().shape(), &[rows, cols]);
        assert_eq!(full.vh.structure().block(0).unwrap().shape(), &[cols, cols]);

        let mut us = vec![0.0; rows * cols];
        for col in 0..cols {
            for inner in 0..rows {
                for row in 0..rows {
                    us[row + rows * col] +=
                        full.u.data()[row + rows * inner] * full.s.data()[inner + rows * col];
                }
            }
        }
        let mut reconstructed = vec![0.0; rows * cols];
        for col in 0..cols {
            for inner in 0..cols {
                for row in 0..rows {
                    reconstructed[row + rows * col] +=
                        us[row + rows * inner] * full.vh.data()[inner + cols * col];
                }
            }
        }
        for (actual, expected) in reconstructed.iter().zip(matrix.data()) {
            assert!((actual - expected).abs() < 1.0e-9);
        }
    }
}

fn assert_checked_pinv_uses_owned_svd_outputs_at_final_gemm<D: crate::factorize::FactorScalar>() {
    let (base, data) = generic_factorization_input();
    let data = data.into_iter().map(D::from_real).collect::<Vec<_>>();
    let (provider, source) = bind_checked_only(&base);
    let input = BoundDynamicTensorRef::try_new(&source, &data).unwrap();
    let output = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        FusionTreeHomSpace::new(
            source.space().homspace().domain().clone(),
            source.space().homspace().codomain().clone(),
        ),
    )
    .unwrap();
    let mut dense = RejectSvdInto::default();

    let result = pinv_direct_into_dyn(&mut dense, &input, output, 0.0).unwrap();

    assert_eq!(dense.svd_into_calls, 0);
    assert_eq!(dense.svd_calls, 2);
    assert_eq!(dense.output_ptrs.len(), 2);
    assert_eq!(dense.gemm_ptrs.len(), 2);
    assert_eq!(
        dense.gemm_ptrs,
        dense
            .output_ptrs
            .iter()
            .map(|&(u, vt)| (vt, u))
            .collect::<Vec<_>>(),
    );
    let regions = source
        .space()
        .structure()
        .coupled_sector_regions(source.space().nout())
        .unwrap()
        .unwrap();
    assert_eq!(dense.gemm_views.len(), regions.len());
    for ((lhs, rhs, lhs_conj, rhs_conj), region) in dense.gemm_views.iter().zip(regions.iter()) {
        let rank = region.rows().min(region.cols());
        assert_eq!(lhs.shape, [region.cols(), rank]);
        assert_eq!(lhs.strides, [rank, 1]);
        assert_eq!(lhs.offset, 0);
        assert_eq!(rhs.shape, [rank, region.rows()]);
        assert_eq!(rhs.strides, [region.rows(), 1]);
        assert_eq!(rhs.offset, 0);
        assert!(*lhs_conj && *rhs_conj);
    }
    assert!(std::ptr::eq(
        result.space().provider_arc().as_ref(),
        provider.as_ref()
    ));
}

#[test]
fn checked_pinv_uses_owned_svd_outputs_at_final_gemm() {
    assert_checked_pinv_uses_owned_svd_outputs_at_final_gemm::<f32>();
    assert_checked_pinv_uses_owned_svd_outputs_at_final_gemm::<f64>();
    assert_checked_pinv_uses_owned_svd_outputs_at_final_gemm::<Complex32>();
    assert_checked_pinv_uses_owned_svd_outputs_at_final_gemm::<Complex64>();
}

#[test]
fn inv_propagates_unsupported_solve_without_svd_fallback() {
    // What: a dense executor without solve support returns its typed capability
    // error instead of silently changing inverse algorithms.
    let tensor = u1_block_endomorphism(&[(0, 1, vec![2.0_f64])]);
    let mut dense = RejectExecutorCalls;
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);

    let error = inv_direct_dyn(&mut dense, &bound.as_ref().dynamic()).unwrap_err();

    assert!(matches!(
        error,
        OperationError::Dense(DenseError::Unsupported {
            op: "solve_into",
            ..
        })
    ));
}

#[test]
fn pinv_adjoint_parent_uses_one_parent_svd_and_the_shared_global_cutoff() {
    // What: the largest singular value is in the later sector, and the first
    // sector sits exactly on the strict global cutoff. The parent-native seam
    // runs one SVD per stored sector, preserves provider identity, and emits
    // the final logical-adjoint orientation directly.
    let canonical = u1_block_endomorphism(&[(0, 1, vec![0.5_f64]), (1, 1, vec![1.0])]);
    let tensor = padded_copy(&U1FusionRule, &canonical);
    let provider = Arc::new(U1FusionRule);
    let bound = bound_tensor(Arc::clone(&provider), &tensor);
    let mut dense = SvdCallSpy::default();
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    crate::factorize::reset_compact_svd_copy_probe();

    let output =
        pinv_adjoint_parent_dyn(&mut dense, &mut context, &bound.as_ref().dynamic(), 0.5).unwrap();
    assert_eq!(dense.svd_calls, 2);
    assert!(Arc::ptr_eq(output.space().provider_arc(), &provider));
    let output: BoundTensorMap<_, _, 1, 1> = typed_from_bound_factor(output).unwrap();
    assert_eq!(scalar_u1_block(output.tensor(), 0), 0.0);
    assert!((scalar_u1_block(output.tensor(), 1) - 1.0).abs() < 1e-12);
    let probe = crate::factorize::compact_svd_copy_probe();
    assert!(probe.input_pack_calls > 0);
    assert!(probe.output_scatter_calls > 0);
}

#[test]
fn pinv_adjoint_parent_rejects_invalid_rcond_before_svd() {
    // What: the hidden seam owns the same validation precedence as ordinary
    // pinv, independently of either facade.
    let tensor = u1_block_endomorphism(&[(0, 1, vec![1.0_f64])]);
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    for rcond in [-1.0, f64::NAN, f64::INFINITY] {
        let mut dense = RejectExecutorCalls;
        let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
        assert!(matches!(
            pinv_adjoint_parent_dyn(&mut dense, &mut context, &bound.as_ref().dynamic(), rcond,),
            Err(OperationError::InvalidArgument { .. })
        ));
    }
}

#[test]
fn pinv_adjoint_parent_discards_unpublished_factors_on_late_svd_failure() {
    // What: a successful first sector cannot publish factors or an output when
    // the second sector's SVD fails.
    let canonical = u1_block_endomorphism(&[(0, 1, vec![2.0_f64]), (1, 1, vec![3.0])]);
    let tensor = padded_copy(&U1FusionRule, &canonical);
    let before = tensor.data().to_vec();
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let mut dense = FailSecondSvd::default();
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    crate::factorize::reset_compact_svd_copy_probe();

    assert!(matches!(
        pinv_adjoint_parent_dyn(&mut dense, &mut context, &bound.as_ref().dynamic(), 0.0,),
        Err(OperationError::Dense(DenseError::Backend {
            op: "svd_into",
            ..
        }))
    ));
    assert_eq!(dense.calls, 2);
    assert_eq!(tensor.data(), before);
    let probe = crate::factorize::compact_svd_copy_probe();
    assert!(probe.input_pack_calls > 0);
    assert!(probe.output_scatter_calls > 0);
}

#[test]
fn polar_validates_every_sector_before_direct_or_fallback_svd_execution() {
    // What: a later invalid sector prevents SVD of an earlier valid sector on both layouts.
    let rule = Z2FusionRule;
    let direct = mixed_rectangular_tensor((4, 2), (1, 3));
    let direct_bound = bound_tensor(Arc::new(rule), &direct);
    assert!(
        crate::factorize::compact_factor_plan_for_test(direct_bound.space())
            .unwrap()
            .is_some()
    );
    let mut dense = SvdCallSpy::default();
    let mut context = default_context();
    let direct_error = left_polar(&mut dense, &mut context, &direct_bound.as_ref()).unwrap_err();
    assert!(matches!(
        direct_error,
        OperationError::InvalidArgument { message }
            if message.contains("left_polar")
                && message.contains("coupled-sector")
    ));
    assert_eq!(dense.svd_calls, 0);

    let fallback_source = mixed_rectangular_tensor((2, 4), (3, 1));
    let fallback_bound = bound_tensor(Arc::new(rule), &fallback_source);
    let fallback_space = fallback_bound.space().adjoint_view().unwrap();
    assert!(
        crate::factorize::compact_factor_plan_for_test(&fallback_space)
            .unwrap()
            .is_none()
    );
    let fallback_input =
        BoundDynamicTensorRef::try_new(&fallback_space, fallback_bound.data()).unwrap();
    let mut dense = SvdCallSpy::default();
    let mut context = default_context();
    let fallback_error = left_polar_dyn(&mut dense, &mut context, &fallback_input).unwrap_err();
    assert!(matches!(
        fallback_error,
        OperationError::InvalidArgument { message }
            if message.contains("left_polar")
                && message.contains("coupled-sector")
    ));
    assert_eq!(dense.svd_calls, 0);
}

#[test]
fn single_precision_svd_and_eig_work_end_to_end() {
    use num_complex::Complex32;
    let rule = Z2FusionRule;
    let sectors = [SectorId::new(0), SectorId::new(1)];
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
    let leg_dim = sectors.len() * degeneracy;
    let homspace = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        )
    };
    let space = || {
        let hom = homspace();
        let key_count = hom.fusion_tree_keys(&rule).len();
        FusionTensorMapSpace::from_degeneracy_shapes_coupled(
            TensorMapSpace::<2, 2>::from_dims([leg_dim, leg_dim], [leg_dim, leg_dim]).unwrap(),
            hom,
            &rule,
            vec![vec![degeneracy; 4]; key_count],
        )
        .unwrap()
    };
    let f32_space = space();
    let len = f32_space.required_len().unwrap();
    let tensor_f32 = TensorMap::<f32, 2, 2>::from_vec_with_fusion_space(
        (0..len)
            .map(|i| ((i * 7 + 3) % 23) as f32 * 0.5 - 5.0)
            .collect(),
        f32_space,
    )
    .unwrap();

    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let svd = svd_compact(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor_f32),
    )
    .unwrap();

    // Reconstruct through an f32 contraction at single precision.
    let mut scaled_vh = svd.vh.tensor().clone();
    {
        let structure = std::sync::Arc::clone(scaled_vh.structure());
        for index in 0..structure.block_count() {
            let block = structure.block(index).unwrap();
            let BlockKey::FusionTree(key) = block.key() else {
                continue;
            };
            let sector = key.codomain_tree().coupled();
            let values = &svd
                .singular_values
                .iter()
                .find(|entry| entry.sector == sector)
                .unwrap()
                .values;
            let shape = block.shape().to_vec();
            let strides = block.strides().to_vec();
            let offset = block.offset();
            let count = shape.iter().product::<usize>();
            let mut indices = vec![0usize; shape.len()];
            for _ in 0..count {
                let position = offset
                    + indices
                        .iter()
                        .zip(&strides)
                        .map(|(&i, &s)| i * s)
                        .sum::<usize>();
                scaled_vh.data_mut()[position] *= values[indices[0]] as f32;
                for axis in 0..shape.len() {
                    indices[axis] += 1;
                    if indices[axis] < shape[axis] {
                        break;
                    }
                    indices[axis] = 0;
                }
            }
        }
    }
    let mut context = TensorContractFusionExecutionContext::<f32, RuleIdentity>::default();
    let reconstructed = crate::compose::compose(&mut context, &rule, &svd.u, &scaled_vh).unwrap();
    let distance = tensor_f32
        .data()
        .iter()
        .zip(reconstructed.data())
        .map(|(lhs, rhs)| ((lhs - rhs) as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(distance < 1e-3, "f32 reconstruction distance {distance}");

    // Complex32 general eigendecomposition returns Complex32 factors.
    let c32_space = space();
    let len = c32_space.required_len().unwrap();
    let tensor_c32 = TensorMap::<Complex32, 2, 2>::from_vec_with_fusion_space(
        (0..len)
            .map(|i| {
                Complex32::new(
                    ((i * 3 + 1) % 13) as f32 - 6.0,
                    ((i * 5 + 2) % 11) as f32 - 5.0,
                )
            })
            .collect(),
        c32_space,
    )
    .unwrap();
    let eig = eig_full(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor_c32),
    )
    .unwrap();
    assert!(!eig.eigenvalues.is_empty());
    for entry in &eig.eigenvalues {
        for pair in entry.values.windows(2) {
            assert!(pair[0].norm() >= pair[1].norm() - 1e-6);
        }
    }
    let _: &TensorMap<Complex32, 2, 1> = &eig.v;
}

#[test]
fn svd_compact_gauge_matches_matrixalgebrakit_phase_rule() {
    use num_complex::Complex64;
    let c = Complex64::new;
    let mut u = vec![
        c(3.0, 4.0),
        c(1.0, -1.0),
        c(-2.0, 0.5),
        c(0.25, -0.5),
        c(-4.0, 0.0),
        c(1.0, 2.0),
    ];
    let mut vh = vec![
        c(0.5, -1.0),
        c(-0.25, 0.75),
        c(1.0, 0.0),
        c(0.0, -2.0),
        c(-1.5, 0.25),
        c(0.75, -0.5),
    ];
    let sigma = [2.0, 0.75];
    let product = |u: &[Complex64], vh: &[Complex64]| -> Vec<Complex64> {
        let mut out = vec![c(0.0, 0.0); 9];
        for col in 0..3 {
            for row in 0..3 {
                for k in 0..2 {
                    out[row + 3 * col] += u[row + 3 * k] * sigma[k] * vh[k + 2 * col];
                }
            }
        }
        out
    };
    let before = product(&u, &vh);
    crate::factorize::svd_compact_gauge(&mut u, 3, 3, &mut vh, 2, 3, 2);
    for &(row, col) in &[(0, 0), (1, 1)] {
        let pivot = u[row + 3 * col];
        assert!(pivot.im.abs() < 1e-14, "pivot {pivot} not real");
        assert!(pivot.re >= 0.0, "pivot {pivot} negative");
    }
    let after = product(&u, &vh);
    for (lhs, rhs) in after.iter().zip(&before) {
        assert!(
            (lhs - rhs).norm() < 1e-13,
            "product changed: {lhs} vs {rhs}"
        );
    }
}

#[test]
fn svd_compact_adjoint_gauge_fixes_final_left_factor() {
    use num_complex::Complex64;
    let c = Complex64::new;
    let mut u = vec![
        c(3.0, 4.0),
        c(1.0, -1.0),
        c(-2.0, 0.5),
        c(0.25, -0.5),
        c(-4.0, 0.0),
        c(1.0, 2.0),
    ];
    let mut vh = vec![
        c(0.5, -1.0),
        c(-0.25, 0.75),
        c(1.0, 0.0),
        c(0.0, -2.0),
        c(-0.5, 1.0),
        c(0.75, -0.5),
    ];
    let sigma = [2.0, 0.75];
    let product = |u: &[Complex64], vh: &[Complex64]| -> Vec<Complex64> {
        let mut out = vec![c(0.0, 0.0); 9];
        for col in 0..3 {
            for row in 0..3 {
                for k in 0..2 {
                    out[row + 3 * col] += u[row + 3 * k] * sigma[k] * vh[k + 2 * col];
                }
            }
        }
        out
    };
    let before = product(&u, &vh);
    crate::factorize::svd_compact_adjoint_gauge(&mut u, 3, 3, &mut vh, 2, 3, 2);

    // These rows become the columns of final U = V after adjointing Vh.
    for &(row, col) in &[(0, 0), (1, 1)] {
        let pivot = vh[row + 2 * col];
        assert!(pivot.im.abs() < 1e-14, "pivot {pivot} not real");
        assert!(pivot.re >= 0.0, "pivot {pivot} negative");
    }
    let after = product(&u, &vh);
    for (lhs, rhs) in after.iter().zip(&before) {
        assert!(
            (lhs - rhs).norm() < 1e-13,
            "product changed: {lhs} vs {rhs}"
        );
    }
}

#[test]
fn svd_full_gauge_fixes_extra_vh_rows_without_changing_product() {
    use num_complex::Complex64;
    let c = Complex64::new;
    let mut u = vec![c(0.0, -2.0), c(0.25, 0.5), c(1.0, -1.0), c(-3.0, 0.0)];
    let mut vh = vec![
        c(1.0, 0.5),
        c(-0.25, 0.75),
        c(1.0, -1.0),
        c(0.5, -0.5),
        c(2.0, 0.0),
        c(-0.5, 0.25),
        c(-1.0, 0.75),
        c(0.0, -1.5),
        c(0.25, 0.0),
    ];
    let sigma = [1.5, 0.7];
    let product = |u: &[Complex64], vh: &[Complex64]| -> Vec<Complex64> {
        let mut out = vec![c(0.0, 0.0); 6];
        for col in 0..3 {
            for row in 0..2 {
                for k in 0..2 {
                    out[row + 2 * col] += u[row + 2 * k] * sigma[k] * vh[k + 3 * col];
                }
            }
        }
        out
    };
    let before = product(&u, &vh);
    crate::factorize::svd_full_gauge(&mut u, 2, 2, &mut vh, 3, 3);
    for &(row, col) in &[(0, 0), (1, 1)] {
        let pivot = u[row + 2 * col];
        assert!(pivot.im.abs() < 1e-14, "U pivot {pivot} not real");
        assert!(pivot.re >= 0.0, "U pivot {pivot} negative");
    }
    let extra_pivot = vh[2]; // row 2, col 0 (row + 3 * col)
    assert!(
        extra_pivot.im.abs() < 1e-14,
        "Vh pivot {extra_pivot} not real"
    );
    assert!(extra_pivot.re >= 0.0, "Vh pivot {extra_pivot} negative");
    let after = product(&u, &vh);
    for (lhs, rhs) in after.iter().zip(&before) {
        assert!(
            (lhs - rhs).norm() < 1e-13,
            "product changed: {lhs} vs {rhs}"
        );
    }
}

#[test]
fn noncanonical_mf_svd_and_eigh_scatter_each_output_block_once() {
    // What: on a four-sector noncanonical MF input, the compact SVD and EIGH
    // fallbacks group each factor side once and iterate only the scattered
    // blocks (F = B = 16 per side), instead of G_s * B = 64 visits per side.
    use crate::factorize::{reset_scatter_visit_probe, scatter_visit_probe, ScatterVisitProbe};
    let sectors = (0..4).map(SectorId::new).collect::<Vec<_>>();
    let rule = || tenet_core::ZNFusionRule::new(4).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let tensor = tsvd_test_tensor(&rule(), &sectors);
    let bound = bound_tensor(Arc::new(rule()), &tensor);
    let adjoint_space = bound.space().adjoint_view().unwrap();
    let input = BoundDynamicTensorRef::try_new(&adjoint_space, bound.data()).unwrap();
    reset_scatter_visit_probe();
    let (u, vh, _) = svd_compact_factors_dyn(&mut dense, &input).unwrap();
    let b_u = u.space().space().structure().block_count();
    let b_vh = vh.space().space().structure().block_count();
    assert_eq!((b_u, b_vh), (16, 16));
    let probe = scatter_visit_probe();
    assert_eq!(
        probe,
        ScatterVisitProbe {
            left_grouped: b_u,
            right_grouped: b_vh,
            left_groups_built: 1,
            right_groups_built: 1,
            left_visits: b_u,
            right_visits: b_vh,
        }
    );
    assert!(probe.left_grouped + probe.left_visits < 4 * b_u);
    assert!(probe.right_grouped + probe.right_visits < 4 * b_vh);

    let tensor = hermitian_test_tensor(&rule(), &sectors);
    let bound = bound_tensor(Arc::new(rule()), &tensor);
    let adjoint_space = bound.space().adjoint_view().unwrap();
    let input = BoundDynamicTensorRef::try_new(&adjoint_space, bound.data()).unwrap();
    reset_scatter_visit_probe();
    let full = eigh_full_dyn(&mut dense, &input).unwrap();
    let b_v = full.v().space().space().structure().block_count();
    assert_eq!(b_v, 16);
    let probe = scatter_visit_probe();
    assert_eq!(
        probe,
        ScatterVisitProbe {
            left_grouped: b_v,
            left_groups_built: 1,
            left_visits: b_v,
            ..ScatterVisitProbe::default()
        }
    );
    assert!(probe.left_grouped + probe.left_visits < 4 * b_v);
}

struct FailSingleLegFold {
    rule: FactorGenericRule,
    fail_at: usize,
    single_leg_folds: Cell<usize>,
}

impl FusionRule for FailSingleLegFold {
    fn rule_identity(&self) -> RuleIdentity {
        self.rule.rule_identity()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        self.rule.fusion_style()
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        self.rule.braiding_style()
    }
    fn vacuum(&self) -> SectorId {
        self.rule.vacuum()
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        self.rule.dual(sector)
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        self.rule.fusion_channels(left, right)
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        self.rule.nsymbol(left, right, coupled)
    }
}

impl CheckedGenericFusion for FailSingleLegFold {
    type Error = LateGenericError;

    fn rule_identity(&self) -> RuleIdentity {
        self.rule.rule_identity()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        self.rule.fusion_style()
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        self.rule.braiding_style()
    }
    fn vacuum(&self) -> SectorId {
        self.rule.vacuum()
    }
    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        Ok(self.rule.dual(sector))
    }
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Ok(self.rule.fusion_channels(left, right))
    }
    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.try_fusion_channels(left, right)
    }
    fn try_coupled_sector_fold(
        &self,
        effective: &[SectorId],
    ) -> Result<CoupledSectorFold, Self::Error> {
        if effective.len() == 1 {
            let call = self.single_leg_folds.get() + 1;
            self.single_leg_folds.set(call);
            if call == self.fail_at {
                return Err(LateGenericError(call));
            }
        }
        Ok(InfallibleGeneric::new(&self.rule)
            .try_coupled_sector_fold(effective)
            .unwrap())
    }
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        Ok(self.rule.nsymbol(left, right, coupled))
    }
}

impl CheckedGenericRigidSymbols for FailSingleLegFold {
    type Scalar = f64;

    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        Ok(self.rule.sqrt_dim_scalar(sector))
    }
    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        Ok(self.rule.inv_sqrt_dim_scalar(sector))
    }
    fn try_frobenius_schur_phase_scalar(
        &self,
        sector: SectorId,
    ) -> Result<Self::Scalar, Self::Error> {
        Ok(self.rule.frobenius_schur_phase_scalar(sector))
    }
    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<Self::Scalar>, Self::Error> {
        Ok(self.rule.f_symbol_generic(a, b, c, d, e, f))
    }
    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        coupled: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, Self::Error> {
        Ok(self.rule.r_symbol_generic(a, b, coupled))
    }
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_svd_compact_preserves_a_diagonal_s_fold_failure() {
    // What: a provider error while laying out checked compact SVD's dense `S`
    // is returned unchanged as `Provider(e)`, after both dense SVDs and with
    // the input untouched. (Formerly asserted through the removed
    // `svd_trunc_factors_dyn_checked_generic`, whose compact call built this
    // `S`; #1534.) U and Vh each enumerate their layout once and fold both
    // one-leg bond sectors (U 1-2, Vh 3-4), so S's first fold is the fifth.
    const FIRST_S_FOLD: usize = 5;
    let (source, data) = generic_factorization_input();
    let failing_provider = Arc::new(FailSingleLegFold {
        rule: FactorGenericRule,
        fail_at: FIRST_S_FOLD,
        single_leg_folds: Cell::new(0),
    });
    let failing_space = BoundDynamicFusionMapSpace::bind_generic(
        source.space().clone(),
        Arc::clone(&failing_provider),
    )
    .unwrap();
    let input = BoundDynamicTensorRef::try_new(&failing_space, &data).unwrap();
    let before = input.data().to_vec();
    let mut dense = CountingDense::default();
    let result = svd_compact_dyn_checked_generic(&mut dense, &input);

    assert!(matches!(
        result,
        Err(CheckedGenericFactorPlanError::Provider(LateGenericError(call)))
            if call == FIRST_S_FOLD
    ));
    assert_eq!(failing_provider.single_leg_folds.get(), FIRST_S_FOLD);
    assert_eq!(dense.svd_calls, 2);
    assert_eq!(dense.svd_into_calls, 0);
    assert_eq!(dense.svd_vals_calls, 0);
    assert_eq!(input.data(), before);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_svd_compact_enumerates_each_factor_layout_once() {
    // What: checked compact SVD queries the provider for exactly one
    // enumeration of each of the U, S and Vh spaces. (The truncating half of
    // the former test went with `svd_trunc_factors_dyn_checked_generic`,
    // #1534.)
    let (source, data) = generic_factorization_input();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), Arc::clone(&provider))
            .unwrap();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = CountingDense::default();
    let Svd { u, s, vh } = svd_compact_dyn_checked_generic(&mut dense, &input).unwrap();
    let compact_calls = provider.calls.get();
    assert!(s.data().len() > 1);
    assert_eq!(
        compact_calls,
        checked_enumeration_calls(&u)
            + checked_enumeration_calls(&s)
            + checked_enumeration_calls(&vh)
    );
}
