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

#[test]
fn op_bearing_batch_applies_rectangular_adjoint_and_alpha_beta() {
    // What: a batch-level Adjoint transposes rectangular parent matrices,
    // conjugates C64 values, and preserves caller alpha/beta accumulation.
    let rows = 2;
    let contracted = 3;
    let cols = 4;
    let lhs = (0..contracted * rows)
        .map(|i| Complex64::new(i as f64 + 1.0, 0.25 * i as f64 - 0.5))
        .collect::<Vec<_>>();
    let rhs = (0..cols * contracted)
        .map(|i| Complex64::new(0.5 * i as f64 - 2.0, 0.75 - 0.1 * i as f64))
        .collect::<Vec<_>>();
    let mut output = vec![Complex64::new(0.5, -0.25); rows * cols];
    let initial = output.clone();
    let alpha = Complex64::new(0.75, -0.5);
    let beta = Complex64::new(-0.25, 0.125);
    let jobs = [DenseGemmBatchJob {
        dst_offset: 0,
        lhs_offset: 0,
        rhs_offset: 0,
        rows,
        contracted,
        cols,
    }];
    let flat_strides = [1];
    let lhs_shape = [lhs.len()];
    let rhs_shape = [rhs.len()];
    let output_shape = [output.len()];
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();
    executor
        .matmul_batch_axpby_with_ops_into(
            DenseWrite::C64(
                DenseViewMut::new(&mut output, &output_shape, &flat_strides, 0).unwrap(),
            ),
            DenseRead::C64(DenseView::new(&lhs, &lhs_shape, &flat_strides, 0).unwrap()),
            DenseRead::C64(DenseView::new(&rhs, &rhs_shape, &flat_strides, 0).unwrap()),
            &jobs,
            &[1],
            MatrixOp::Adjoint,
            MatrixOp::Adjoint,
            DenseScalar::C64(alpha),
            DenseScalar::C64(beta),
        )
        .unwrap();

    for col in 0..cols {
        for row in 0..rows {
            let mut sum = Complex64::new(0.0, 0.0);
            for inner in 0..contracted {
                let left = lhs[inner + contracted * row].conj();
                let right = rhs[col + cols * inner].conj();
                sum += left * right;
            }
            let index = row + rows * col;
            assert_c64_close(output[index], alpha * sum + beta * initial[index], 1.0e-12);
        }
    }
}

#[test]
fn op_bearing_batch_rejects_offset_overflow_before_view_construction() {
    // What: malformed public batch jobs return a typed offset error instead of
    // wrapping an operand base offset before the transposed view is validated.
    let lhs = vec![1.0, 2.0];
    let rhs = vec![3.0];
    let mut output = vec![0.0];
    let jobs = [DenseGemmBatchJob {
        dst_offset: 0,
        lhs_offset: usize::MAX,
        rhs_offset: 0,
        rows: 1,
        contracted: 1,
        cols: 1,
    }];
    let shape = [1];
    let strides = [1];
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();

    let error = executor
        .matmul_batch_axpby_with_ops_into(
            DenseWrite::F64(DenseViewMut::new(&mut output, &shape, &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&lhs, &shape, &strides, 1).unwrap()),
            DenseRead::F64(DenseView::new(&rhs, &shape, &strides, 0).unwrap()),
            &jobs,
            &[1],
            MatrixOp::Adjoint,
            MatrixOp::Identity,
            DenseScalar::F64(1.0),
            DenseScalar::F64(0.0),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        DenseError::OffsetOverflow { value: usize::MAX }
    ));
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
    // Four same-shape constant-stride jobs form one length-4 run (>=
    // STRIDED_RUN_MIN), so the batch routes through the strided-batch seam as a
    // single dispatch rather than one call per job.
    let mut lhs = Vec::new();
    let mut rhs = Vec::new();
    for block in 0..4 {
        let base = block as f64;
        lhs.extend_from_slice(&[1.0 + base, 2.0 + base, 3.0 + base, 4.0 + base]);
        rhs.extend_from_slice(&[5.0 + base, 6.0 + base, 7.0 + base, 8.0 + base]);
    }
    let mut output = vec![-99.0; 4 * 4];
    let jobs = [0usize, 1, 2, 3]
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
    let runs = strided_batch_runs(&jobs);
    assert_eq!(runs, vec![4]);
    let flat_shape = [4 * 4];
    let flat_strides = [1usize];

    let mut executor = DefaultDenseExecutor::new();
    executor.reset_seam_dispatches();
    executor
        .matmul_batch_axpby_into(
            DenseWrite::F64(DenseViewMut::new(&mut output, &flat_shape, &flat_strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&lhs, &flat_shape, &flat_strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&rhs, &flat_shape, &flat_strides, 0).unwrap()),
            &jobs,
            &runs,
            DenseScalar::F64(1.0),
            DenseScalar::F64(0.0),
        )
        .unwrap();

    assert_eq!(
        executor.seam_dispatches(),
        1,
        "same-shape strided batch made {} seam dispatches for {} jobs",
        executor.seam_dispatches(),
        jobs.len()
    );
    assert!(
        executor.seam_dispatches() < jobs.len(),
        "batched GEMM seam dispatch count must not scale with same-shape job count"
    );
    for block in 0..4 {
        let start = block * 4;
        let expected = matmul_f64(&lhs[start..start + 4], &rhs[start..start + 4], 2, 2, 2);
        for (actual, expected) in output[start..start + 4].iter().zip(expected) {
            assert_f64_close(*actual, expected, 1.0e-12);
        }
    }

    let lhs_f32 = lhs.iter().map(|&value| value as f32).collect::<Vec<_>>();
    let rhs_f32 = rhs.iter().map(|&value| value as f32).collect::<Vec<_>>();
    let mut output_f32 = vec![-99.0_f32; 4 * 4];
    let mut executor = DefaultDenseExecutor::new();
    executor
        .matmul_batch_axpby_into(
            DenseWrite::F32(
                DenseViewMut::new(&mut output_f32, &flat_shape, &flat_strides, 0).unwrap(),
            ),
            DenseRead::F32(DenseView::new(&lhs_f32, &flat_shape, &flat_strides, 0).unwrap()),
            DenseRead::F32(DenseView::new(&rhs_f32, &flat_shape, &flat_strides, 0).unwrap()),
            &jobs,
            &runs,
            DenseScalar::F32(1.0),
            DenseScalar::F32(0.0),
        )
        .unwrap();
    assert_eq!(executor.seam_dispatches(), 1);
    for block in 0..4 {
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
    let mut output_c32 = vec![Complex32::new(-99.0, -99.0); 4 * 4];
    let mut executor = DefaultDenseExecutor::new();
    executor
        .matmul_batch_axpby_into(
            DenseWrite::C32(
                DenseViewMut::new(&mut output_c32, &flat_shape, &flat_strides, 0).unwrap(),
            ),
            DenseRead::C32(DenseView::new(&lhs_c32, &flat_shape, &flat_strides, 0).unwrap()),
            DenseRead::C32(DenseView::new(&rhs_c32, &flat_shape, &flat_strides, 0).unwrap()),
            &jobs,
            &runs,
            DenseScalar::C32(Complex32::new(1.0, 0.0)),
            DenseScalar::C32(Complex32::new(0.0, 0.0)),
        )
        .unwrap();
    assert_eq!(executor.seam_dispatches(), 1);
    for block in 0..4 {
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
    let mut output_c64 = vec![Complex64::new(-99.0, -99.0); 4 * 4];
    let mut executor = DefaultDenseExecutor::new();
    executor
        .matmul_batch_axpby_into(
            DenseWrite::C64(
                DenseViewMut::new(&mut output_c64, &flat_shape, &flat_strides, 0).unwrap(),
            ),
            DenseRead::C64(DenseView::new(&lhs_c64, &flat_shape, &flat_strides, 0).unwrap()),
            DenseRead::C64(DenseView::new(&rhs_c64, &flat_shape, &flat_strides, 0).unwrap()),
            &jobs,
            &runs,
            DenseScalar::C64(Complex64::new(1.0, 0.0)),
            DenseScalar::C64(Complex64::new(0.0, 0.0)),
        )
        .unwrap();
    assert_eq!(executor.seam_dispatches(), 1);
    for block in 0..4 {
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
fn default_executor_bundles_short_runs_into_one_seam_dispatch() {
    // Structural guard for issue #103: a fragmented batch (three runs — two
    // length-2 same-shape runs plus a singleton, all below STRIDED_RUN_MIN)
    // must dispatch ONE grouped seam call, not one per run. Seam-call count
    // stays flat as a batch fragments into more runs.
    //
    // Layout: shape A = 2x2x2 (blocks at storage offsets 0,4), shape B = 1x3x1
    // (blocks at 8,10), shape C = 2x1x2 singleton (at 12). All dst ranges
    // disjoint. lhs/rhs share the same flat buffer regions via the offsets.
    let jobs = vec![
        DenseGemmBatchJob {
            dst_offset: 0,
            lhs_offset: 0,
            rhs_offset: 0,
            rows: 2,
            contracted: 2,
            cols: 2,
        },
        DenseGemmBatchJob {
            dst_offset: 4,
            lhs_offset: 4,
            rhs_offset: 4,
            rows: 2,
            contracted: 2,
            cols: 2,
        },
        DenseGemmBatchJob {
            dst_offset: 8,
            lhs_offset: 8,
            rhs_offset: 8,
            rows: 1,
            contracted: 3,
            cols: 1,
        },
        DenseGemmBatchJob {
            dst_offset: 9,
            lhs_offset: 11,
            rhs_offset: 11,
            rows: 1,
            contracted: 3,
            cols: 1,
        },
        DenseGemmBatchJob {
            dst_offset: 10,
            lhs_offset: 14,
            rhs_offset: 14,
            rows: 2,
            contracted: 1,
            cols: 2,
        },
    ];
    let runs = strided_batch_runs(&jobs);
    assert_eq!(
        runs,
        vec![2, 2, 1],
        "batch must present three runs, none >= cutoff"
    );

    // Storage large enough for every lhs/rhs/dst range referenced above.
    let buf_len = 16usize;
    let lhs: Vec<f64> = (0..buf_len).map(|i| 1.0 + i as f64).collect();
    let rhs: Vec<f64> = (0..buf_len).map(|i| 2.0 + 0.5 * i as f64).collect();
    let mut output = vec![-99.0; buf_len];
    let flat_shape = [buf_len];
    let flat_strides = [1usize];

    let mut executor = DefaultDenseExecutor::new();
    executor.reset_seam_dispatches();
    executor
        .matmul_batch_axpby_into(
            DenseWrite::F64(DenseViewMut::new(&mut output, &flat_shape, &flat_strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&lhs, &flat_shape, &flat_strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&rhs, &flat_shape, &flat_strides, 0).unwrap()),
            &jobs,
            &runs,
            DenseScalar::F64(1.0),
            DenseScalar::F64(0.0),
        )
        .unwrap();

    assert_eq!(
        executor.seam_dispatches(),
        1,
        "three sub-cutoff runs must bundle into one grouped seam dispatch, got {}",
        executor.seam_dispatches()
    );
    assert!(
        executor.seam_dispatches() < runs.len(),
        "seam dispatch count must not scale with the number of runs"
    );

    // Byte-for-byte correctness of every bundled job.
    for job in &jobs {
        let lhs_block = &lhs[job.lhs_offset..job.lhs_offset + job.rows * job.contracted];
        let rhs_block = &rhs[job.rhs_offset..job.rhs_offset + job.contracted * job.cols];
        let expected = matmul_f64(lhs_block, rhs_block, job.rows, job.contracted, job.cols);
        let got = &output[job.dst_offset..job.dst_offset + job.rows * job.cols];
        for (actual, expected) in got.iter().zip(expected) {
            assert_f64_close(*actual, expected, 1.0e-12);
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

#[cfg(all(feature = "cpu-faer", not(feature = "cpu-blas")))]
#[test]
fn faer_only_build_rejects_uncompiled_blas_provider() {
    let error = DefaultDenseExecutor::with_kind(CpuBackendKind::Blas).unwrap_err();
    assert!(error.to_string().contains("cpu-blas"));
}

#[cfg(all(feature = "cpu-blas", not(feature = "cpu-faer")))]
#[test]
fn blas_only_build_rejects_uncompiled_faer_provider() {
    let error = DefaultDenseExecutor::with_kind(CpuBackendKind::Faer).unwrap_err();
    assert!(error.to_string().contains("cpu-faer"));
}

// Exercises the values-only trait *defaults* (full decomposition minus the
// vectors). `DefaultDenseExecutor` overrides them, so this wraps it in an
// executor that implements svd/eigh/eig but leaves svd_vals/eigh_vals/eig_vals
// at their trait default, then checks the fallback spectra agree with the
// backend's no-vector override to LAPACK precision.
#[test]
fn values_only_defaults_fall_back_to_the_full_decomposition_spectrum() {
    struct FullOnly(DefaultDenseExecutor);
    impl DenseExecutor for FullOnly {
        fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            self.0.svd(input)
        }
        fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            self.0.qr(input)
        }
        fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            self.0.eigh(input)
        }
        fn eig(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            self.0.eig(input)
        }
        fn dot_general_into(
            &mut self,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
            config: &DenseDotConfig,
        ) -> Result<(), DenseError> {
            self.0.dot_general_into(output, lhs, rhs, config)
        }
    }

    let shape = [2usize, 2];
    let strides = [1usize, 2]; // column-major
    let m = vec![2.0f64, 1.0, 1.0, 3.0]; // symmetric, so eigh applies too
    fn view<'a>(data: &'a [f64], shape: &'a [usize], strides: &'a [usize]) -> DenseRead<'a> {
        DenseRead::F64(DenseView::new(data, shape, strides, 0).unwrap())
    }

    let mut fallback = FullOnly(DefaultDenseExecutor::new());
    let mut direct = DefaultDenseExecutor::new();
    let tol = 1e-10;

    let f = fallback.svd_vals(view(&m, &shape, &strides)).unwrap();
    let d = direct.svd_vals(view(&m, &shape, &strides)).unwrap();
    let (f, d) = (f.as_f64_slice().unwrap(), d.as_f64_slice().unwrap());
    assert_eq!(f.len(), 2);
    for (a, b) in f.iter().zip(d) {
        assert_f64_close(*a, *b, tol);
    }

    let f = fallback.eigh_vals(view(&m, &shape, &strides)).unwrap();
    let d = direct.eigh_vals(view(&m, &shape, &strides)).unwrap();
    let (f, d) = (f.as_f64_slice().unwrap(), d.as_f64_slice().unwrap());
    assert_eq!(f.len(), 2);
    for (a, b) in f.iter().zip(d) {
        assert_f64_close(*a, *b, tol);
    }

    let f = fallback.eig_vals(view(&m, &shape, &strides)).unwrap();
    let d = direct.eig_vals(view(&m, &shape, &strides)).unwrap();
    let (f, d) = (f.as_c64_slice().unwrap(), d.as_c64_slice().unwrap());
    assert_eq!(f.len(), 2);
    for (a, b) in f.iter().zip(d) {
        assert_c64_close(*a, *b, tol);
    }
}

fn batch_job(shape: (usize, usize, usize), offsets: (usize, usize, usize)) -> DenseGemmBatchJob {
    DenseGemmBatchJob {
        rows: shape.0,
        contracted: shape.1,
        cols: shape.2,
        dst_offset: offsets.0,
        lhs_offset: offsets.1,
        rhs_offset: offsets.2,
    }
}

#[test]
fn strided_batch_runs_partitions_same_shape_constant_stride_runs() {
    // Two length-2 constant-stride runs (shapes A, B) followed by a singleton
    // (shape C): the plan-time partition the executor routes over.
    let jobs = vec![
        batch_job((2, 2, 2), (0, 0, 0)),
        batch_job((2, 2, 2), (4, 4, 4)),
        batch_job((3, 1, 2), (8, 8, 8)),
        batch_job((3, 1, 2), (14, 11, 10)),
        batch_job((1, 5, 1), (20, 14, 12)),
    ];
    assert_eq!(strided_batch_runs(&jobs), vec![2, 2, 1]);
    // Empty batch => empty partition; the lengths always cover every job.
    assert_eq!(strided_batch_runs(&[]), Vec::<usize>::new());
    assert_eq!(
        strided_batch_runs(&jobs).iter().sum::<usize>(),
        jobs.len(),
        "run partition must cover all jobs"
    );
}

#[test]
fn strided_batch_runs_breaks_on_shape_and_stride_changes() {
    // A shape change ends a run; a non-constant stride within one shape also
    // ends it (the second/third jobs share a shape but not a common stride).
    let jobs = vec![
        batch_job((2, 2, 2), (0, 0, 0)),
        batch_job((2, 2, 2), (4, 4, 4)),
        batch_job((2, 2, 2), (100, 4, 4)),
    ];
    assert_eq!(strided_batch_runs(&jobs), vec![2, 1]);
}
