#![allow(dead_code)]

use super::*;
use num_complex::{Complex32, Complex64};

fn assert_f64_close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {expected}, got {actual}, tol={tol}"
    );
}

fn assert_f32_close(actual: f32, expected: f32, tol: f32) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {expected}, got {actual}, tol={tol}"
    );
}

fn assert_c32_close(actual: Complex32, expected: Complex32, tol: f32) {
    assert_f32_close(actual.re, expected.re, tol);
    assert_f32_close(actual.im, expected.im, tol);
}

fn assert_c64_close(actual: Complex64, expected: Complex64, tol: f64) {
    assert_f64_close(actual.re, expected.re, tol);
    assert_f64_close(actual.im, expected.im, tol);
}

// Regression guard for the conjugated-contraction fast path: a conj flag on
// `dot_general_into` must fold conjugation into the kernel and produce exactly
// what contracting an elementwise-conjugated operand would — and it must
// actually change the result (so the flag can't be silently dropped back to a
// no-op or a bypassed scalar loop).
#[test]
fn dot_general_conjugation_flag_matches_materialized_conjugate() {
    let c = |re: f64, im: f64| Complex64::new(re, im);
    let shape = [2usize, 2];
    let strides = [1usize, 2]; // column-major
    let lhs = vec![c(1.0, 1.0), c(3.0, 2.0), c(2.0, -1.0), c(4.0, -3.0)];
    let rhs = vec![c(5.0, -2.0), c(7.0, -4.0), c(6.0, 1.0), c(8.0, 2.0)];

    let run = |lhs_data: &[Complex64], lhs_conj: bool, rhs_conj: bool| -> Vec<Complex64> {
        let mut out = vec![c(0.0, 0.0); 4];
        let mut executor = DefaultDenseExecutor::new();
        executor
            .dot_general_into(
                DenseWrite::C64(DenseViewMut::new(&mut out, &shape, &strides, 0).unwrap()),
                DenseRead::C64(DenseView::new(lhs_data, &shape, &strides, 0).unwrap()),
                DenseRead::C64(DenseView::new(&rhs, &shape, &strides, 0).unwrap()),
                &DenseDotConfig::matmul().with_conjugation(lhs_conj, rhs_conj),
            )
            .unwrap();
        out
    };

    let via_flag = run(&lhs, true, false);
    let lhs_conjugated: Vec<Complex64> = lhs.iter().map(|z| z.conj()).collect();
    let via_materialized = run(&lhs_conjugated, false, false);
    for (actual, expected) in via_flag.iter().zip(&via_materialized) {
        assert_c64_close(*actual, *expected, 1.0e-12);
    }

    let plain = run(&lhs, false, false);
    assert!(
        via_flag
            .iter()
            .zip(&plain)
            .any(|(a, b)| (a - b).norm() > 1.0e-9),
        "conjugation flag had no effect on the result"
    );
}

fn col_major_index(rows: usize, row: usize, col: usize) -> usize {
    row + col * rows
}

fn transpose_f32(mat: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)];
        }
    }
    out
}

fn transpose_f64(mat: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)];
        }
    }
    out
}

fn transpose_c32(mat: &[Complex32], rows: usize, cols: usize) -> Vec<Complex32> {
    let mut out = vec![Complex32::new(0.0, 0.0); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)];
        }
    }
    out
}

fn transpose_c64(mat: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)];
        }
    }
    out
}

fn conjugate_transpose_c32(mat: &[Complex32], rows: usize, cols: usize) -> Vec<Complex32> {
    let mut out = vec![Complex32::new(0.0, 0.0); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)].conj();
        }
    }
    out
}

fn conjugate_transpose_c64(mat: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)].conj();
        }
    }
    out
}

fn matmul_f32(lhs: &[f32], rhs: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0; m * n];
    for j in 0..n {
        for p in 0..k {
            let rhs_pj = rhs[col_major_index(k, p, j)];
            for i in 0..m {
                out[col_major_index(m, i, j)] += lhs[col_major_index(m, i, p)] * rhs_pj;
            }
        }
    }
    out
}

fn matmul_f64(lhs: &[f64], rhs: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for j in 0..n {
        for p in 0..k {
            let rhs_pj = rhs[col_major_index(k, p, j)];
            for i in 0..m {
                out[col_major_index(m, i, j)] += lhs[col_major_index(m, i, p)] * rhs_pj;
            }
        }
    }
    out
}

fn matmul_c32(
    lhs: &[Complex32],
    rhs: &[Complex32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Complex32> {
    let mut out = vec![Complex32::new(0.0, 0.0); m * n];
    for j in 0..n {
        for p in 0..k {
            let rhs_pj = rhs[col_major_index(k, p, j)];
            for i in 0..m {
                out[col_major_index(m, i, j)] += lhs[col_major_index(m, i, p)] * rhs_pj;
            }
        }
    }
    out
}

fn matmul_c64(
    lhs: &[Complex64],
    rhs: &[Complex64],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); m * n];
    for j in 0..n {
        for p in 0..k {
            let rhs_pj = rhs[col_major_index(k, p, j)];
            for i in 0..m {
                out[col_major_index(m, i, j)] += lhs[col_major_index(m, i, p)] * rhs_pj;
            }
        }
    }
    out
}

fn diag_f32(values: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; values.len() * values.len()];
    for (i, value) in values.iter().enumerate() {
        out[col_major_index(values.len(), i, i)] = *value;
    }
    out
}

fn diag_f64(values: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; values.len() * values.len()];
    for (i, value) in values.iter().enumerate() {
        out[col_major_index(values.len(), i, i)] = *value;
    }
    out
}

fn diag_c32_from_real(values: &[f32]) -> Vec<Complex32> {
    let mut out = vec![Complex32::new(0.0, 0.0); values.len() * values.len()];
    for (i, value) in values.iter().enumerate() {
        out[col_major_index(values.len(), i, i)] = Complex32::new(*value, 0.0);
    }
    out
}

fn diag_c64_from_real(values: &[f64]) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); values.len() * values.len()];
    for (i, value) in values.iter().enumerate() {
        out[col_major_index(values.len(), i, i)] = Complex64::new(*value, 0.0);
    }
    out
}

#[test]
fn dense_view_rejects_out_of_bounds_layout() {
    let data = [0.0; 6];
    let shape = [2, 3];
    let strides = [1, 4];
    let err = DenseView::new(&data, &shape, &strides, 0).unwrap_err();
    assert_eq!(err, DenseError::OutOfBounds);
}

#[cfg(feature = "tenferro")]
#[test]
fn default_executor_matmul_into_matches_tensorkit_recoupling_view_for_all_gemm_dtypes() {
    let lhs_shape = [2, 3];
    let lhs_strides = [1, 2];
    let rhs_shape = [3, 2];
    let rhs_strides = [1, 3];
    let out_shape = [2, 2];
    let out_strides = [1, 4];
    let out_offset = 1;

    let mut executor = DefaultDenseExecutor::new();

    let lhs_f32 = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let u_f32 = vec![10.0_f32, 100.0, 1000.0, 20.0, 200.0, 2000.0];
    let mut out_f32 = vec![-1.0_f32; 8];
    executor
        .matmul_into(
            DenseWrite::F32(
                DenseViewMut::new(&mut out_f32, &out_shape, &out_strides, out_offset).unwrap(),
            ),
            DenseRead::F32(DenseView::new(&lhs_f32, &lhs_shape, &lhs_strides, 0).unwrap()),
            DenseRead::F32(DenseView::new(&u_f32, &rhs_shape, &rhs_strides, 0).unwrap()),
        )
        .unwrap();
    assert_eq!(
        out_f32,
        vec![-1.0, 5310.0, 6420.0, -1.0, -1.0, 10620.0, 12840.0, -1.0]
    );

    let lhs_f64 = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let u_f64 = vec![10.0_f64, 100.0, 1000.0, 20.0, 200.0, 2000.0];
    let mut out_f64 = vec![-1.0_f64; 8];
    executor
        .matmul_into(
            DenseWrite::F64(
                DenseViewMut::new(&mut out_f64, &out_shape, &out_strides, out_offset).unwrap(),
            ),
            DenseRead::F64(DenseView::new(&lhs_f64, &lhs_shape, &lhs_strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&u_f64, &rhs_shape, &rhs_strides, 0).unwrap()),
        )
        .unwrap();
    assert_eq!(
        out_f64,
        vec![-1.0, 5310.0, 6420.0, -1.0, -1.0, 10620.0, 12840.0, -1.0]
    );

    let lhs_c32 = lhs_f32
        .iter()
        .map(|&value| Complex32::new(value, 0.0))
        .collect::<Vec<_>>();
    let u_c32 = u_f32
        .iter()
        .map(|&value| Complex32::new(value, 0.0))
        .collect::<Vec<_>>();
    let mut out_c32 = vec![Complex32::new(-1.0, -2.0); 8];
    executor
        .matmul_into(
            DenseWrite::C32(
                DenseViewMut::new(&mut out_c32, &out_shape, &out_strides, out_offset).unwrap(),
            ),
            DenseRead::C32(DenseView::new(&lhs_c32, &lhs_shape, &lhs_strides, 0).unwrap()),
            DenseRead::C32(DenseView::new(&u_c32, &rhs_shape, &rhs_strides, 0).unwrap()),
        )
        .unwrap();
    assert_eq!(
        out_c32,
        vec![
            Complex32::new(-1.0, -2.0),
            Complex32::new(5310.0, 0.0),
            Complex32::new(6420.0, 0.0),
            Complex32::new(-1.0, -2.0),
            Complex32::new(-1.0, -2.0),
            Complex32::new(10620.0, 0.0),
            Complex32::new(12840.0, 0.0),
            Complex32::new(-1.0, -2.0),
        ]
    );

    let lhs_c64 = lhs_f64
        .iter()
        .map(|&value| Complex64::new(value, 0.0))
        .collect::<Vec<_>>();
    let u_c64 = u_f64
        .iter()
        .map(|&value| Complex64::new(value, 0.0))
        .collect::<Vec<_>>();
    let mut out_c64 = vec![Complex64::new(-1.0, -2.0); 8];
    executor
        .matmul_into(
            DenseWrite::C64(
                DenseViewMut::new(&mut out_c64, &out_shape, &out_strides, out_offset).unwrap(),
            ),
            DenseRead::C64(DenseView::new(&lhs_c64, &lhs_shape, &lhs_strides, 0).unwrap()),
            DenseRead::C64(DenseView::new(&u_c64, &rhs_shape, &rhs_strides, 0).unwrap()),
        )
        .unwrap();
    assert_eq!(
        out_c64,
        vec![
            Complex64::new(-1.0, -2.0),
            Complex64::new(5310.0, 0.0),
            Complex64::new(6420.0, 0.0),
            Complex64::new(-1.0, -2.0),
            Complex64::new(-1.0, -2.0),
            Complex64::new(10620.0, 0.0),
            Complex64::new(12840.0, 0.0),
            Complex64::new(-1.0, -2.0),
        ]
    );
}

#[cfg(feature = "tenferro")]
#[test]
fn default_executor_fuses_same_shape_strided_batch_jobs_for_all_gemm_dtypes() {
    let mut lhs = Vec::new();
    let mut rhs = Vec::new();
    for block in 0..2 {
        let base = block as f64;
        lhs.extend_from_slice(&[1.0 + base, 2.0 + base, 3.0 + base, 4.0 + base]);
        rhs.extend_from_slice(&[5.0 + base, 6.0 + base, 7.0 + base, 8.0 + base]);
    }
    let mut output = vec![-99.0; 2 * 4];
    let jobs = [0usize, 1]
        .into_iter()
        .map(|block| DenseGemmBatchJob {
            dst_offset: block * 4,
            lhs_offset: block * 4,
            rhs_offset: block * 4,
            rows: 2,
            contracted: 2,
            cols: 2,
        })
        .collect::<Vec<_>>();
    let flat_shape = [2 * 4];
    let flat_strides = [1usize];

    let mut executor = DefaultDenseExecutor::new();
    executor.reset_logical_gemm_dispatches();
    executor
        .matmul_batch_axpby_into(
            DenseWrite::F64(DenseViewMut::new(&mut output, &flat_shape, &flat_strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&lhs, &flat_shape, &flat_strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&rhs, &flat_shape, &flat_strides, 0).unwrap()),
            &jobs,
            DenseScalar::F64(1.0),
            DenseScalar::F64(0.0),
        )
        .unwrap();

    assert_eq!(
        executor.logical_gemm_dispatches(),
        1,
        "same-shape strided batch submitted {} logical GEMM dispatches for {} jobs",
        executor.logical_gemm_dispatches(),
        jobs.len()
    );
    assert!(
        executor.logical_gemm_dispatches() < jobs.len(),
        "batched GEMM logical dispatch count must not scale with same-shape job count"
    );
    for block in 0..2 {
        let start = block * 4;
        let expected = matmul_f64(&lhs[start..start + 4], &rhs[start..start + 4], 2, 2, 2);
        for (actual, expected) in output[start..start + 4].iter().zip(expected) {
            assert_f64_close(*actual, expected, 1.0e-12);
        }
    }

    let lhs_f32 = lhs.iter().map(|&value| value as f32).collect::<Vec<_>>();
    let rhs_f32 = rhs.iter().map(|&value| value as f32).collect::<Vec<_>>();
    let mut output_f32 = vec![-99.0_f32; 2 * 4];
    let mut executor = DefaultDenseExecutor::new();
    executor
        .matmul_batch_axpby_into(
            DenseWrite::F32(
                DenseViewMut::new(&mut output_f32, &flat_shape, &flat_strides, 0).unwrap(),
            ),
            DenseRead::F32(DenseView::new(&lhs_f32, &flat_shape, &flat_strides, 0).unwrap()),
            DenseRead::F32(DenseView::new(&rhs_f32, &flat_shape, &flat_strides, 0).unwrap()),
            &jobs,
            DenseScalar::F32(1.0),
            DenseScalar::F32(0.0),
        )
        .unwrap();
    assert_eq!(executor.logical_gemm_dispatches(), 1);
    for block in 0..2 {
        let start = block * 4;
        let expected = matmul_f32(
            &lhs_f32[start..start + 4],
            &rhs_f32[start..start + 4],
            2,
            2,
            2,
        );
        for (actual, expected) in output_f32[start..start + 4].iter().zip(expected) {
            assert_f32_close(*actual, expected, 1.0e-4);
        }
    }

    let lhs_c32 = lhs_f32
        .iter()
        .map(|&value| Complex32::new(value, 0.25 * value))
        .collect::<Vec<_>>();
    let rhs_c32 = rhs_f32
        .iter()
        .map(|&value| Complex32::new(value, -0.125 * value))
        .collect::<Vec<_>>();
    let mut output_c32 = vec![Complex32::new(-99.0, -99.0); 2 * 4];
    let mut executor = DefaultDenseExecutor::new();
    executor
        .matmul_batch_axpby_into(
            DenseWrite::C32(
                DenseViewMut::new(&mut output_c32, &flat_shape, &flat_strides, 0).unwrap(),
            ),
            DenseRead::C32(DenseView::new(&lhs_c32, &flat_shape, &flat_strides, 0).unwrap()),
            DenseRead::C32(DenseView::new(&rhs_c32, &flat_shape, &flat_strides, 0).unwrap()),
            &jobs,
            DenseScalar::C32(Complex32::new(1.0, 0.0)),
            DenseScalar::C32(Complex32::new(0.0, 0.0)),
        )
        .unwrap();
    assert_eq!(executor.logical_gemm_dispatches(), 1);
    for block in 0..2 {
        let start = block * 4;
        let expected = matmul_c32(
            &lhs_c32[start..start + 4],
            &rhs_c32[start..start + 4],
            2,
            2,
            2,
        );
        for (actual, expected) in output_c32[start..start + 4].iter().zip(expected) {
            assert_c32_close(*actual, expected, 1.0e-3);
        }
    }

    let lhs_c64 = lhs
        .iter()
        .map(|&value| Complex64::new(value, 0.25 * value))
        .collect::<Vec<_>>();
    let rhs_c64 = rhs
        .iter()
        .map(|&value| Complex64::new(value, -0.125 * value))
        .collect::<Vec<_>>();
    let mut output_c64 = vec![Complex64::new(-99.0, -99.0); 2 * 4];
    let mut executor = DefaultDenseExecutor::new();
    executor
        .matmul_batch_axpby_into(
            DenseWrite::C64(
                DenseViewMut::new(&mut output_c64, &flat_shape, &flat_strides, 0).unwrap(),
            ),
            DenseRead::C64(DenseView::new(&lhs_c64, &flat_shape, &flat_strides, 0).unwrap()),
            DenseRead::C64(DenseView::new(&rhs_c64, &flat_shape, &flat_strides, 0).unwrap()),
            &jobs,
            DenseScalar::C64(Complex64::new(1.0, 0.0)),
            DenseScalar::C64(Complex64::new(0.0, 0.0)),
        )
        .unwrap();
    assert_eq!(executor.logical_gemm_dispatches(), 1);
    for block in 0..2 {
        let start = block * 4;
        let expected = matmul_c64(
            &lhs_c64[start..start + 4],
            &rhs_c64[start..start + 4],
            2,
            2,
            2,
        );
        for (actual, expected) in output_c64[start..start + 4].iter().zip(expected) {
            assert_c64_close(*actual, expected, 1.0e-12);
        }
    }
}

#[cfg(feature = "tenferro")]
#[test]
fn default_executor_qr_reads_transposed_views_for_all_linalg_dtypes() {
    let f32_data = vec![1.0_f32, -2.0, 3.0, 0.5, -1.0, 4.0];
    let f64_data = vec![1.0_f64, -2.0, 3.0, 0.5, -1.0, 4.0];
    let c32_data = vec![
        Complex32::new(1.0, 0.5),
        Complex32::new(-2.0, 1.0),
        Complex32::new(3.0, -0.25),
        Complex32::new(0.5, -1.0),
        Complex32::new(-1.0, 0.75),
        Complex32::new(4.0, 1.5),
    ];
    let c64_data = vec![
        Complex64::new(1.0, 0.5),
        Complex64::new(-2.0, 1.0),
        Complex64::new(3.0, -0.25),
        Complex64::new(0.5, -1.0),
        Complex64::new(-1.0, 0.75),
        Complex64::new(4.0, 1.5),
    ];
    let shape = [3, 2];
    let strides = [2, 1];
    let mut executor = DefaultDenseExecutor::new();

    let outputs = executor
        .qr(DenseRead::F32(
            DenseView::new(&f32_data, &shape, &strides, 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(outputs[0].dtype(), DenseDType::F32);
    let recon = matmul_f32(
        outputs[0].as_f32_slice().unwrap(),
        outputs[1].as_f32_slice().unwrap(),
        3,
        2,
        2,
    );
    let expected = transpose_f32(&f32_data, 2, 3);
    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_f32_close(*actual, *expected, 1.0e-5);
    }

    let outputs = executor
        .qr(DenseRead::F64(
            DenseView::new(&f64_data, &shape, &strides, 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(outputs[0].dtype(), DenseDType::F64);
    let recon = matmul_f64(
        outputs[0].as_f64_slice().unwrap(),
        outputs[1].as_f64_slice().unwrap(),
        3,
        2,
        2,
    );
    let expected = transpose_f64(&f64_data, 2, 3);
    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_f64_close(*actual, *expected, 1.0e-9);
    }

    let outputs = executor
        .qr(DenseRead::C32(
            DenseView::new(&c32_data, &shape, &strides, 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(outputs[0].dtype(), DenseDType::C32);
    let recon = matmul_c32(
        outputs[0].as_c32_slice().unwrap(),
        outputs[1].as_c32_slice().unwrap(),
        3,
        2,
        2,
    );
    let expected = transpose_c32(&c32_data, 2, 3);
    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_c32_close(*actual, *expected, 1.0e-5);
    }

    let outputs = executor
        .qr(DenseRead::C64(
            DenseView::new(&c64_data, &shape, &strides, 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(outputs[0].dtype(), DenseDType::C64);
    let recon = matmul_c64(
        outputs[0].as_c64_slice().unwrap(),
        outputs[1].as_c64_slice().unwrap(),
        3,
        2,
        2,
    );
    let expected = transpose_c64(&c64_data, 2, 3);
    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_c64_close(*actual, *expected, 1.0e-9);
    }
}

#[cfg(feature = "tenferro")]
#[test]
fn default_executor_eigh_reads_transposed_views_for_all_linalg_dtypes() {
    let f32_data = vec![4.0_f32, 1.0, 1.0, 3.0];
    let f64_data = vec![4.0_f64, 1.0, 1.0, 3.0];
    let c32_data = vec![
        Complex32::new(4.0, 0.0),
        Complex32::new(1.0, -0.5),
        Complex32::new(1.0, 0.5),
        Complex32::new(3.0, 0.0),
    ];
    let c64_data = vec![
        Complex64::new(4.0, 0.0),
        Complex64::new(1.0, -0.5),
        Complex64::new(1.0, 0.5),
        Complex64::new(3.0, 0.0),
    ];
    let shape = [2, 2];
    let strides = [2, 1];
    let mut executor = DefaultDenseExecutor::new();

    let outputs = executor
        .eigh(DenseRead::F32(
            DenseView::new(&f32_data, &shape, &strides, 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(outputs[0].dtype(), DenseDType::F32);
    assert_eq!(outputs[1].dtype(), DenseDType::F32);
    let values = outputs[0].as_f32_slice().unwrap();
    let vectors = outputs[1].as_f32_slice().unwrap();
    let recon = matmul_f32(
        &matmul_f32(vectors, &diag_f32(values), 2, 2, 2),
        &transpose_f32(vectors, 2, 2),
        2,
        2,
        2,
    );
    let expected = transpose_f32(&f32_data, 2, 2);
    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_f32_close(*actual, *expected, 1.0e-5);
    }

    let outputs = executor
        .eigh(DenseRead::F64(
            DenseView::new(&f64_data, &shape, &strides, 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(outputs[0].dtype(), DenseDType::F64);
    assert_eq!(outputs[1].dtype(), DenseDType::F64);
    let values = outputs[0].as_f64_slice().unwrap();
    let vectors = outputs[1].as_f64_slice().unwrap();
    let recon = matmul_f64(
        &matmul_f64(vectors, &diag_f64(values), 2, 2, 2),
        &transpose_f64(vectors, 2, 2),
        2,
        2,
        2,
    );
    let expected = transpose_f64(&f64_data, 2, 2);
    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_f64_close(*actual, *expected, 1.0e-10);
    }

    let outputs = executor
        .eigh(DenseRead::C32(
            DenseView::new(&c32_data, &shape, &strides, 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(outputs[0].dtype(), DenseDType::F32);
    assert_eq!(outputs[1].dtype(), DenseDType::C32);
    let values = outputs[0].as_f32_slice().unwrap();
    let vectors = outputs[1].as_c32_slice().unwrap();
    let recon = matmul_c32(
        &matmul_c32(vectors, &diag_c32_from_real(values), 2, 2, 2),
        &conjugate_transpose_c32(vectors, 2, 2),
        2,
        2,
        2,
    );
    let expected = transpose_c32(&c32_data, 2, 2);
    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_c32_close(*actual, *expected, 1.0e-5);
    }

    let outputs = executor
        .eigh(DenseRead::C64(
            DenseView::new(&c64_data, &shape, &strides, 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(outputs[0].dtype(), DenseDType::F64);
    assert_eq!(outputs[1].dtype(), DenseDType::C64);
    let values = outputs[0].as_f64_slice().unwrap();
    let vectors = outputs[1].as_c64_slice().unwrap();
    let recon = matmul_c64(
        &matmul_c64(vectors, &diag_c64_from_real(values), 2, 2, 2),
        &conjugate_transpose_c64(vectors, 2, 2),
        2,
        2,
        2,
    );
    let expected = transpose_c64(&c64_data, 2, 2);
    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_c64_close(*actual, *expected, 1.0e-10);
    }
}

#[cfg(feature = "tenferro")]
#[test]
fn default_executor_svd_accepts_all_supported_linalg_dtypes() {
    let f32_data = [1.0_f32, -2.0, 0.5, 4.0];
    let f64_data = [1.0_f64, -2.0, 0.5, 4.0];
    let c32_data = [
        Complex32::new(1.0, 0.5),
        Complex32::new(-2.0, 1.0),
        Complex32::new(0.5, -0.25),
        Complex32::new(4.0, 1.5),
    ];
    let c64_data = [
        Complex64::new(1.0, 0.5),
        Complex64::new(-2.0, 1.0),
        Complex64::new(0.5, -0.25),
        Complex64::new(4.0, 1.5),
    ];
    let shape = [2, 2];
    let strides = [2, 1];

    let mut executor = DefaultDenseExecutor::new();
    for (input, dtype) in [
        (
            DenseRead::F32(DenseView::new(&f32_data, &shape, &strides, 0).unwrap()),
            DenseDType::F32,
        ),
        (
            DenseRead::F64(DenseView::new(&f64_data, &shape, &strides, 0).unwrap()),
            DenseDType::F64,
        ),
        (
            DenseRead::C32(DenseView::new(&c32_data, &shape, &strides, 0).unwrap()),
            DenseDType::C32,
        ),
        (
            DenseRead::C64(DenseView::new(&c64_data, &shape, &strides, 0).unwrap()),
            DenseDType::C64,
        ),
    ] {
        let outputs = executor.svd(input).unwrap();
        assert_eq!(outputs[0].dtype(), dtype);
        assert!(matches!(
            (dtype, outputs[1].dtype()),
            (DenseDType::F32, DenseDType::F32)
                | (DenseDType::F64, DenseDType::F64)
                | (DenseDType::C32, DenseDType::F32)
                | (DenseDType::C64, DenseDType::F64)
        ));
        assert_eq!(outputs[2].dtype(), dtype);
    }
}

#[cfg(feature = "tenferro")]
#[test]
fn default_executor_svd_into_writes_strided_destination_views() {
    let data = [1.0_f64, -2.0, 0.5, 4.0];
    let input_shape = [2, 2];
    let input_strides = [1, 2];
    let input = DenseRead::F64(DenseView::new(&data, &input_shape, &input_strides, 0).unwrap());

    let mut executor = DefaultDenseExecutor::new();
    let expected = executor.svd(input).unwrap();

    let mut u = vec![-99.0; 8];
    let mut s = vec![-99.0; 4];
    let mut vt = vec![-99.0; 8];
    let matrix_shape = [2, 2];
    let matrix_strides = [1, 3];
    let s_shape = [2];
    let s_strides = [2];
    executor
        .svd_into(
            input,
            DenseWrite::F64(DenseViewMut::new(&mut u, &matrix_shape, &matrix_strides, 1).unwrap()),
            DenseWrite::F64(DenseViewMut::new(&mut s, &s_shape, &s_strides, 0).unwrap()),
            DenseWrite::F64(DenseViewMut::new(&mut vt, &matrix_shape, &matrix_strides, 1).unwrap()),
        )
        .unwrap();

    let expected_u = expected[0].as_f64_slice().unwrap();
    let expected_s = expected[1].as_f64_slice().unwrap();
    let expected_vt = expected[2].as_f64_slice().unwrap();
    for col in 0..2 {
        for row in 0..2 {
            assert_f64_close(u[1 + row + 3 * col], expected_u[row + 2 * col], 1e-12);
            assert_f64_close(vt[1 + row + 3 * col], expected_vt[row + 2 * col], 1e-12);
        }
    }
    for index in 0..2 {
        assert_f64_close(s[2 * index], expected_s[index], 1e-12);
    }
}

#[cfg(feature = "tenferro")]
#[test]
fn default_executor_qr_into_writes_strided_destination_views() {
    let data = [
        Complex64::new(1.0, 0.5),
        Complex64::new(-2.0, 1.0),
        Complex64::new(0.5, -0.25),
        Complex64::new(4.0, 1.5),
    ];
    let input_shape = [2, 2];
    let input_strides = [1, 2];
    let input = DenseRead::C64(DenseView::new(&data, &input_shape, &input_strides, 0).unwrap());

    let mut executor = DefaultDenseExecutor::new();
    let expected = executor.qr(input).unwrap();

    let sentinel = Complex64::new(-99.0, 0.0);
    let mut q = vec![sentinel; 8];
    let mut r = vec![sentinel; 8];
    let matrix_shape = [2, 2];
    let matrix_strides = [1, 3];
    executor
        .qr_into(
            input,
            DenseWrite::C64(DenseViewMut::new(&mut q, &matrix_shape, &matrix_strides, 1).unwrap()),
            DenseWrite::C64(DenseViewMut::new(&mut r, &matrix_shape, &matrix_strides, 1).unwrap()),
        )
        .unwrap();

    let expected_q = expected[0].as_c64_slice().unwrap();
    let expected_r = expected[1].as_c64_slice().unwrap();
    for col in 0..2 {
        for row in 0..2 {
            assert_c64_close(q[1 + row + 3 * col], expected_q[row + 2 * col], 1e-12);
            assert_c64_close(r[1 + row + 3 * col], expected_r[row + 2 * col], 1e-12);
        }
    }
}

#[cfg(feature = "tenferro")]
#[test]
fn default_executor_eigh_into_writes_strided_destination_views() {
    let data = [
        Complex64::new(4.0, 0.0),
        Complex64::new(1.0, 0.5),
        Complex64::new(1.0, -0.5),
        Complex64::new(3.0, 0.0),
    ];
    let input_shape = [2, 2];
    let input_strides = [1, 2];
    let input = DenseRead::C64(DenseView::new(&data, &input_shape, &input_strides, 0).unwrap());

    let mut executor = DefaultDenseExecutor::new();
    let expected = executor.eigh(input).unwrap();

    let mut values = vec![-99.0; 4];
    let sentinel = Complex64::new(-99.0, 0.0);
    let mut vectors = vec![sentinel; 8];
    let values_shape = [2];
    let values_strides = [2];
    let matrix_shape = [2, 2];
    let matrix_strides = [1, 3];
    executor
        .eigh_into(
            input,
            DenseWrite::F64(
                DenseViewMut::new(&mut values, &values_shape, &values_strides, 1).unwrap(),
            ),
            DenseWrite::C64(
                DenseViewMut::new(&mut vectors, &matrix_shape, &matrix_strides, 1).unwrap(),
            ),
        )
        .unwrap();

    let expected_values = expected[0].as_f64_slice().unwrap();
    let expected_vectors = expected[1].as_c64_slice().unwrap();
    for index in 0..2 {
        assert_f64_close(values[1 + 2 * index], expected_values[index], 1e-12);
    }
    for col in 0..2 {
        for row in 0..2 {
            assert_c64_close(
                vectors[1 + row + 3 * col],
                expected_vectors[row + 2 * col],
                1e-12,
            );
        }
    }
}

#[cfg(feature = "tenferro")]
#[test]
fn default_executor_rejects_integer_linalg_view() {
    let data = [1_i32, 0, 0, 1];
    let shape = [2, 2];
    let strides = [1, 2];
    let view = DenseView::new(&data, &shape, &strides, 0).unwrap();

    let mut executor = DefaultDenseExecutor::new();
    let err = executor.qr(DenseRead::I32(view)).unwrap_err();

    assert!(matches!(
        err,
        DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "qr_read",
            ref message,
        } if message.contains("unsupported dtype")
    ));
}
