use super::*;

/// Tests one packed CUDA matrix region with the host EIGH rule; the
/// one-region case of [`cuda_hermitian_regions`]. Test support only: no
/// production caller decides a single region (#1805).
fn cuda_is_hermitian_region<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    offset: usize,
    n: usize,
) -> Result<bool, DenseError> {
    Ok(cuda_hermitian_regions::<D>(
        ctx,
        src,
        &[(offset, n)],
        <D::Real as CudaRealScalar>::EPSILON.powf(0.75),
    )?[0])
}

#[test]
fn rectangular_operand_views_use_parent_native_strides() {
    assert_eq!(cuda_operand_view(MatrixOp::Identity, 2, 3), ([1, 2], false));
    assert_eq!(cuda_operand_view(MatrixOp::Adjoint, 2, 3), ([3, 1], true));
    assert_eq!(cuda_operand_view(MatrixOp::Identity, 3, 4), ([1, 3], false));
    assert_eq!(cuda_operand_view(MatrixOp::Adjoint, 3, 4), ([4, 1], true));
}

#[test]
fn a_zero_descriptor_scale_is_rejected_before_any_device_work() {
    // What: `alpha = 0` is the one scale whose backend behaviour is not
    // `alpha * src` — the source read may be skipped, erasing NaN/Inf — so
    // it is a typed rejection rather than a silently different answer. The
    // check needs no device, which is what makes it precede every upload.
    for alpha in [0.0_f64, -0.0] {
        let err = reject_zero_alpha::<f64>("cuda_region_axpby", alpha)
            .expect_err("a zero descriptor scale must be rejected");
        assert!(
            matches!(
                err,
                DenseError::Unsupported {
                    op: "cuda_region_axpby",
                    ..
                }
            ),
            "{err}"
        );
        assert!(
            err.to_string().contains("CudaRegionCoefficient::Zero"),
            "{err}"
        );
    }
    let complex_zero = Complex64::new(-0.0, 0.0);
    assert!(reject_zero_alpha::<Complex64>("cuda_region_axpby", complex_zero).is_err());

    // Every other scale passes, including the non-finite ones: they are
    // ordinary multiplications.
    for alpha in [1.0_f64, -2.5, f64::NAN, f64::INFINITY] {
        assert!(reject_zero_alpha::<f64>("cuda_region_axpby", alpha).is_ok());
    }
    assert!(reject_zero_alpha::<Complex64>("cuda_region_axpby", Complex64::new(0.0, -1.0)).is_ok());
}

#[test]
fn ensure_cuda_device_accepts_matching_operands() {
    assert!(
        ensure_cuda_device(0, "op", &[("a", 0), ("b", 0)]).is_ok(),
        "operands on the context device must be accepted"
    );
}

#[test]
fn ensure_cuda_device_rejects_a_foreign_operand() {
    let err = ensure_cuda_device(0, "cuda_matmul", &[("lhs", 0), ("rhs", 1)])
        .expect_err("an operand on another device must be rejected");
    match err {
        DenseError::Backend {
            backend,
            op,
            message,
        } => {
            assert_eq!(backend, DenseBackend::Cuda);
            assert_eq!(op, "cuda_matmul");
            assert!(
                message.contains("rhs"),
                "message names the operand: {message}"
            );
            assert!(
                message.contains("device 1"),
                "message names the device: {message}"
            );
        }
        other => panic!("expected a CUDA backend error, got {other:?}"),
    }
}

/// A spectrum produced on `device`, of `len` values (host data stands in for
/// the device buffer: the guard reads only the recorded device).
fn spectrum_on(device: usize, len: usize) -> CudaSpectrum {
    CudaSpectrum {
        tensor: Tensor::from_vec_col_major(vec![len], vec![1.0_f64; len]).unwrap(),
        device,
    }
}

fn assert_foreign_spectrum(err: DenseError, op: &str, operand: &str) {
    match err {
        DenseError::Backend {
            backend: DenseBackend::Cuda,
            op: got,
            message,
        } => {
            assert_eq!(got, op);
            assert!(message.contains(operand), "names the operand: {message}");
            assert!(message.contains("device 1"), "names the device: {message}");
        }
        other => panic!("expected a CUDA device mismatch, got {other:?}"),
    }
}

#[test]
fn the_spectrum_device_guard_rejects_any_foreign_spectrum_even_an_empty_one() {
    assert!(ensure_spectra_device(0, "op", &[]).is_ok());
    assert!(ensure_spectra_device(0, "op", &[spectrum_on(0, 3), spectrum_on(0, 0)]).is_ok());
    let err = ensure_spectra_device(0, "op", &[spectrum_on(0, 3), spectrum_on(1, 0)])
        .expect_err("an empty spectrum from device 1 must be rejected");
    assert_foreign_spectrum(err, "op", "spectra[1]");
}

#[test]
#[ignore = "requires a real CUDA device"]
fn every_spectrum_consumer_rejects_a_foreign_spectrum_before_any_device_work() {
    // What: a spectrum recorded on device 1, read through a device-0 context,
    // is rejected by each consumer before it submits or allocates anything —
    // empty ones too, which otherwise skip all device work. One GPU suffices:
    // the guard reads the recorded device, so relabelling stands in for a
    // second producer (the two-device test below uses a real one).
    let mut ctx = CudaDenseContext::new(0).unwrap();
    let src = CudaDenseStorage::upload::<f64>(&ctx, &[2.0, 0.5, 0.5, 1.0]).unwrap();
    let foreign = |ctx: &mut CudaDenseContext, n: usize| {
        let (mut values, _) = cuda_eigh_region::<f64>(ctx, &src, 0, n).unwrap();
        values.device = 1;
        values
    };
    let batched_foreign = |ctx: &mut CudaDenseContext| {
        let (mut values, _) = cuda_eigh_region_batched::<f64>(ctx, &src, 0, 2, 1, 4).unwrap();
        values.device = 1;
        values
    };
    let mut dst =
        CudaDenseStorage::upload::<Complex64>(&ctx, &[Complex64::new(7.0, 0.0); 4]).unwrap();
    for len in [2, 0] {
        let local = || spectrum_on(0, 0);
        let spectra = [
            local(),
            if len == 0 {
                spectrum_on(1, 0)
            } else {
                foreign(&mut ctx, 2)
            },
        ];
        let before = cuda_transfer_stats();
        let err = cuda_download_spectra::<f64>(&mut ctx, &spectra).unwrap_err();
        assert_foreign_spectrum(err, "cuda_download_spectra", "spectra[1]");
        assert_eq!(cuda_transfer_stats(), before, "download, len {len}");

        let spectra = [
            local(),
            if len == 0 {
                spectrum_on(1, 0)
            } else {
                batched_foreign(&mut ctx)
            },
        ];
        let before = cuda_transfer_stats();
        let err = cuda_download_batched_spectra::<f64>(&mut ctx, &spectra, 1).unwrap_err();
        assert_foreign_spectrum(err, "cuda_download_spectra", "spectra[1]");
        assert_eq!(cuda_transfer_stats(), before, "batched download, len {len}");

        let spectrum = if len == 0 {
            spectrum_on(1, 0)
        } else {
            foreign(&mut ctx, 2)
        };
        let before = cuda_transfer_stats();
        let err =
            cuda_copy_spectrum_into::<Complex64>(&mut ctx, spectrum, &mut dst, 0, 3).unwrap_err();
        assert_foreign_spectrum(err, "cuda_copy_spectrum", "spectrum");
        assert_eq!(cuda_transfer_stats(), before, "copy, len {len}");
    }
    assert_eq!(
        dst.download::<Complex64>(&ctx).unwrap(),
        vec![Complex64::new(7.0, 0.0); 4],
        "a rejected copy writes nothing"
    );
}

#[test]
#[ignore = "requires two idle CUDA devices"]
fn a_spectrum_from_another_device_is_rejected_by_every_consumer() {
    let mut producer = CudaDenseContext::new(1).unwrap();
    let mut consumer = CudaDenseContext::new(0).unwrap();
    let src = CudaDenseStorage::upload::<f64>(&producer, &[2.0, 0.5, 0.5, 1.0]).unwrap();
    let mut eigh = || {
        cuda_eigh_region::<f64>(&mut producer, &src, 0, 2)
            .unwrap()
            .0
    };
    let (values, copied) = (eigh(), eigh());
    let (_, singular, _) = cuda_svd_region::<f64>(&mut producer, &src, 0, 2, 2).unwrap();
    let (batched, _) = cuda_eigh_region_batched::<f64>(&mut producer, &src, 0, 2, 1, 4).unwrap();
    assert_eq!(
        (
            values.device,
            copied.device,
            singular.device,
            batched.device
        ),
        (1, 1, 1, 1)
    );
    let mut dst = CudaDenseStorage::upload::<f64>(&consumer, &[7.0; 4]).unwrap();

    let before = cuda_transfer_stats();
    let err = cuda_download_spectra::<f64>(&mut consumer, &[values, singular]).unwrap_err();
    assert_foreign_spectrum(err, "cuda_download_spectra", "spectra[0]");
    let err = cuda_download_batched_spectra::<f64>(&mut consumer, &[batched], 1).unwrap_err();
    assert_foreign_spectrum(err, "cuda_download_spectra", "spectra[0]");
    let err = cuda_copy_spectrum_into::<f64>(&mut consumer, copied, &mut dst, 0, 3).unwrap_err();
    assert_foreign_spectrum(err, "cuda_copy_spectrum", "spectrum");
    assert_eq!(
        cuda_transfer_stats(),
        before,
        "no consumer submits anything"
    );
    assert_eq!(dst.download::<f64>(&consumer).unwrap(), vec![7.0; 4]);
}

#[test]
fn svd_factor_shape_contract_covers_rectangular_and_bad_backend_results() {
    assert!(validate_svd_factor_shapes(&[4, 3], 3, &[3, 3], 4, 3).is_ok());
    assert!(validate_svd_factor_shapes(&[3, 3], 3, &[3, 4], 3, 4).is_ok());
    assert!(validate_svd_factor_shapes(&[4, 4], 3, &[3, 3], 4, 3).is_err());
    assert!(validate_svd_factor_shapes(&[4, 3], 2, &[3, 3], 4, 3).is_err());
    assert!(validate_svd_factor_shapes(&[4, 3], 3, &[4, 3], 4, 3).is_err());
}

#[test]
fn qr_factor_shapes_must_match_the_requested_compact_problem() {
    assert!(validate_qr_factor_shapes(&[4, 3], &[3, 3], 4, 3).is_ok());
    assert!(validate_qr_factor_shapes(&[4, 4], &[3, 3], 4, 3).is_err());
    assert!(validate_qr_factor_shapes(&[4, 3], &[4, 3], 4, 3).is_err());
}

#[test]
fn eigh_factor_shapes_must_match_the_requested_square_problem() {
    assert!(validate_eigh_factor_shapes(3, &[3, 3], 3).is_ok());
    assert!(validate_eigh_factor_shapes(2, &[3, 3], 3).is_err());
    assert!(validate_eigh_factor_shapes(3, &[3, 2], 3).is_err());
}

/// Each admitted dtype owns its own scalar-operand slot, so a context used
/// from two dtypes never hands one of them the other's payload-typed `1`.
#[test]
fn every_admitted_dtype_owns_a_distinct_operand_slot() {
    let slots = [
        f32::OPERAND_SLOT,
        f64::OPERAND_SLOT,
        Complex32::OPERAND_SLOT,
        Complex64::OPERAND_SLOT,
    ];
    for (index, slot) in slots.iter().enumerate() {
        assert!(*slot < SCALAR_OPERAND_SLOTS);
        assert!(
            !slots[..index].contains(slot),
            "slot {slot} is claimed twice"
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_bad_later_gather_row_rejects_the_call_before_any_write() {
    // What: with a valid first entry and an out-of-range later one, the
    // gather is rejected and the destination keeps every sentinel; the
    // valid list alone places the selected columns.
    let mut ctx = CudaDenseContext::new(0).unwrap();
    let (n, members) = (3, 2);
    let mut data = vec![0.0; n * n * members];
    for member in 0..members {
        for i in 0..n {
            data[member * n * n + i + n * i] = (i + 1 + member) as f64;
        }
    }
    let src = CudaDenseStorage::upload_owned(&ctx, data).unwrap();
    let (_, vectors) =
        cuda_eigh_region_batched::<f64>(&mut ctx, &src, 0, n, members, n * n).unwrap();
    let reference = vectors.download::<f64>(&ctx).unwrap();
    let sentinel = vec![7.0; n * n * members];
    let mut dst = CudaDenseStorage::upload(&ctx, &sentinel).unwrap();
    let columns = [2, 0, 1, 1, 2, 0];
    let copies_before = cuda_transfer_stats().copy_calls;
    let bad = cuda_gather_columns_batched_into::<f64>(
        &mut ctx,
        &mut dst,
        0,
        n,
        n * n,
        &vectors,
        n,
        members,
        &columns,
        &[(0, 0, 1), (1, 2, 2)],
    );
    assert!(bad.is_err());
    assert_eq!(
        cuda_transfer_stats().copy_calls,
        copies_before,
        "no submission"
    );
    assert_eq!(dst.download::<f64>(&ctx).unwrap(), sentinel);

    cuda_gather_columns_batched_into::<f64>(
        &mut ctx,
        &mut dst,
        0,
        n,
        n * n,
        &vectors,
        n,
        members,
        &columns,
        &[(0, 0, 1), (1, 1, 2)],
    )
    .unwrap();
    let got = dst.download::<f64>(&ctx).unwrap();
    for member in 0..members {
        for (c, &column) in columns[member * n..][..n].iter().enumerate() {
            for row in 0..n {
                let want = reference[member * n * n + row + n * column];
                assert_eq!(got[member * n * n + row + n * c], want);
            }
        }
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn an_empty_spectrum_copy_touches_nothing_even_for_a_complex_payload() {
    let mut ctx = CudaDenseContext::new(0).unwrap();
    let empty = CudaDenseStorage::upload_owned::<f64>(&ctx, Vec::new()).unwrap();
    let spectrum = CudaSpectrum {
        tensor: empty.tensor,
        device: ctx.device,
    };
    let values = vec![Complex64::new(1.0, 2.0); 4];
    let mut dst = CudaDenseStorage::upload::<Complex64>(&ctx, &values).unwrap();
    let before = cuda_transfer_stats();
    cuda_copy_spectrum_into::<Complex64>(&mut ctx, spectrum, &mut dst, 0, 3).unwrap();
    assert_eq!(cuda_transfer_stats(), before, "no cast, copy or transfer");
    assert_eq!(dst.download::<Complex64>(&ctx).unwrap(), values);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn cuda_hermitian_region_is_scaled_and_downloads_only_scalar_metadata() {
    let mut ctx = CudaDenseContext::new(0).unwrap();
    let n = 4;
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        data[i + n * i] = 1.0;
    }
    data[n] = 32.0 * f64::EPSILON;
    data[1] = data[n];
    let storage = CudaDenseStorage::upload::<f64>(&ctx, &data).unwrap();

    CUDA_FULL_DOWNLOAD_BYTES.with(|cell| cell.set(0));
    CUDA_METADATA_DOWNLOAD_BYTES.with(|cell| cell.set(0));
    assert!(cuda_is_hermitian_region::<f64>(&mut ctx, &storage, 0, n).unwrap());
    assert_eq!(CUDA_FULL_DOWNLOAD_BYTES.with(Cell::get), 0);
    assert!(CUDA_METADATA_DOWNLOAD_BYTES.with(Cell::get) <= 4 * 8);

    let near_threshold = |ctx: &CudaDenseContext, delta: f64| {
        // For [[1, delta], [0, 1]], the shared half-residual rule changes
        // truth value at delta = 128 eps up to negligible O(delta^2).
        CudaDenseStorage::upload::<f64>(ctx, &[1.0, 0.0, delta, 1.0]).unwrap()
    };
    let below = near_threshold(&ctx, 120.0 * f64::EPSILON);
    let above = near_threshold(&ctx, 136.0 * f64::EPSILON);
    assert!(cuda_is_hermitian_region::<f64>(&mut ctx, &below, 0, 2).unwrap());
    assert!(!cuda_is_hermitian_region::<f64>(&mut ctx, &above, 0, 2).unwrap());

    let zero = CudaDenseStorage::upload::<f64>(&ctx, &vec![0.0; n * n]).unwrap();
    assert!(cuda_is_hermitian_region::<f64>(&mut ctx, &zero, 0, n).unwrap());

    data[n] = 256.0 * f64::EPSILON;
    let asymmetric = CudaDenseStorage::upload::<f64>(&ctx, &data).unwrap();
    assert!(!cuda_is_hermitian_region::<f64>(&mut ctx, &asymmetric, 0, n).unwrap());

    for scale in [f64::from_bits(0x0010_0000_0000_0000), 2.0_f64.powi(500)] {
        let scaled: Vec<_> = data.iter().map(|value| value * scale).collect();
        let scaled = CudaDenseStorage::upload::<f64>(&ctx, &scaled).unwrap();
        assert!(!cuda_is_hermitian_region::<f64>(&mut ctx, &scaled, 0, n).unwrap());
    }

    for bad in [f64::NAN, f64::INFINITY] {
        let mut nonfinite = vec![0.0; n * n];
        nonfinite[0] = bad;
        let nonfinite = CudaDenseStorage::upload::<f64>(&ctx, &nonfinite).unwrap();
        assert!(!cuda_is_hermitian_region::<f64>(&mut ctx, &nonfinite, 0, n).unwrap());
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn cuda_transfer_bytes_scale_with_the_payload_dtype() {
    // #1268: a complex payload must move the same *number* of buffers as
    // the real one and exactly `size_of::<Complex64>() / size_of::<f64>()`
    // times the bytes for the same element count.
    let ctx = CudaDenseContext::new(0).unwrap();
    let elements = 16;
    let real: Vec<f64> = (0..elements).map(|index| index as f64).collect();
    let complex: Vec<Complex64> = (0..elements)
        .map(|index| Complex64::new(index as f64, -(index as f64) - 0.5))
        .collect();

    CUDA_FULL_DOWNLOAD_BYTES.with(|cell| cell.set(0));
    let real_device = CudaDenseStorage::upload::<f64>(&ctx, &real).unwrap();
    assert_eq!(real_device.dtype(), DenseDType::F64);
    assert_eq!(real_device.download::<f64>(&ctx).unwrap(), real);
    let real_bytes = CUDA_FULL_DOWNLOAD_BYTES.with(|cell| cell.replace(0));

    let complex_device = CudaDenseStorage::upload::<Complex64>(&ctx, &complex).unwrap();
    assert_eq!(complex_device.dtype(), DenseDType::C64);
    assert_eq!(complex_device.download::<Complex64>(&ctx).unwrap(), complex);
    let complex_bytes = CUDA_FULL_DOWNLOAD_BYTES.with(|cell| cell.replace(0));

    assert_eq!(real_bytes, elements * std::mem::size_of::<f64>());
    assert_eq!(
        complex_bytes,
        real_bytes * std::mem::size_of::<Complex64>() / std::mem::size_of::<f64>()
    );

    // A dtype mismatch is a typed error, never a reinterpretation.
    assert!(matches!(
        complex_device.download::<f64>(&ctx),
        Err(DenseError::DTypeMismatch {
            expected: DenseDType::F64,
            actual: DenseDType::C64,
            ..
        })
    ));
    assert!(matches!(
        real_device.region_view::<Complex64>(4, 4, 4, 0),
        Err(DenseError::DTypeMismatch {
            expected: DenseDType::C64,
            actual: DenseDType::F64,
            ..
        })
    ));
}

#[test]
#[ignore = "requires a real CUDA device"]
fn cuda_transfer_counters_attribute_one_upload_download_and_gemm() {
    let mut ctx = CudaDenseContext::new(0).unwrap();
    let n = 2;
    let lhs_host = vec![1.0_f64, 2.0, 3.0, 4.0];
    let rhs_host = vec![5.0_f64, 6.0, 7.0, 8.0];

    reset_cuda_transfer_stats();
    CUDA_FULL_DOWNLOAD_BYTES.with(|cell| cell.set(0));
    let lhs = CudaDenseStorage::upload::<f64>(&ctx, &lhs_host).unwrap();
    let rhs = CudaDenseStorage::upload::<f64>(&ctx, &rhs_host).unwrap();
    let mut dst = CudaDenseStorage::upload::<f64>(&ctx, &vec![0.0_f64; n * n]).unwrap();
    let after_uploads = cuda_transfer_stats();
    assert_eq!(after_uploads.h2d_calls, 3);
    assert_eq!(
        after_uploads.h2d_bytes,
        (3 * n * n * std::mem::size_of::<f64>()) as u64
    );
    assert_eq!(after_uploads.device_allocs, 3);
    assert_eq!(after_uploads.d2h_calls, 0);
    assert_eq!(after_uploads.gemm_calls, 0);

    cuda_gemm_region_into::<f64>(
        &mut ctx, &mut dst, 0, n, &lhs, 0, n, &rhs, 0, n, n, n, n, 1.0, 0.0,
    )
    .unwrap();
    let after_gemm = cuda_transfer_stats();
    assert_eq!(after_gemm.gemm_calls, 1);
    assert_eq!(after_gemm.h2d_calls, after_uploads.h2d_calls);
    assert_eq!(after_gemm.d2h_calls, 0);

    let values = dst.download::<f64>(&ctx).unwrap();
    let after_download = cuda_transfer_stats();
    assert_eq!(after_download.d2h_calls, 1);
    assert_eq!(
        after_download.d2h_bytes,
        (n * n * std::mem::size_of::<f64>()) as u64
    );
    // Column-major 2x2 product, so the counters above describe a real GEMM.
    assert_eq!(values, vec![23.0, 34.0, 31.0, 46.0]);

    // The finer-grained test counter stays consistent with the always
    // compiled one for the same download.
    assert_eq!(
        CUDA_FULL_DOWNLOAD_BYTES.with(|cell| cell.replace(0)) as u64,
        after_download.d2h_bytes
    );

    reset_cuda_transfer_stats();
    assert_eq!(cuda_transfer_stats(), CudaTransferStats::default());
}

#[test]
#[ignore = "requires a real CUDA device"]
fn warm_up_costs_one_gemm_one_solver_call_and_a_bounded_fixed_traffic() {
    // A context built directly here has never submitted work: only
    // `RuntimeBuilder::build` warms one, so this is a cold backend.
    let mut ctx = CudaDenseContext::new(0).unwrap();

    reset_cuda_transfer_stats();
    ctx.warm_up().unwrap();
    let after = cuda_transfer_stats();
    // The documented fixed cost of the warm-up, as the rustdoc states it.
    // `device_allocs` counts the three 1-element uploads plus the
    // eigenvector factor; all four are local to `warm_up` and dropped
    // before it returns, so nothing of this survives construction. There is
    // no live/peak device-buffer observation at this seam to assert on.
    assert_eq!(
        after,
        CudaTransferStats {
            h2d_calls: 3,
            h2d_bytes: 3 * std::mem::size_of::<f64>() as u64,
            d2h_calls: 1,
            d2h_bytes: std::mem::size_of::<f64>() as u64,
            device_allocs: 4,
            gemm_calls: 1,
            solver_calls: 1,
            copy_calls: 0,
            gauge_ops: 0,
        }
    );

    // Warming an already warm context repeats the same bounded work rather
    // than growing with the number of calls.
    reset_cuda_transfer_stats();
    ctx.warm_up().unwrap();
    assert_eq!(cuda_transfer_stats(), after);
    reset_cuda_transfer_stats();
}

#[test]
fn cuda_errors_are_labelled_as_the_cuda_backend() {
    // Regression for #38: CUDA failures must not be reported as Tenferro.
    let err = cuda_error("cuda_svd", "boom");
    let text = err.to_string();
    assert!(text.contains("Cuda"), "formatted error names CUDA: {text}");
    assert!(
        !text.contains("Tenferro"),
        "must not mislabel as Tenferro: {text}"
    );
}
