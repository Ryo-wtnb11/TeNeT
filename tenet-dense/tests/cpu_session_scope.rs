//! S1a evidence: `DefaultDenseExecutor` opens one Tenferro CPU session per
//! dense phase, and the phase's results are unchanged.
//!
//! These live in their own test binary on purpose: `cpu_session_stats` is a
//! process-wide counter, so a delta is only meaningful when no unrelated test
//! opens a session in the same process. Inside this binary the mutex below
//! serializes the readers.

#![cfg(all(feature = "tenferro", not(feature = "provider-inject")))]

use num_complex::Complex64;
use tenet_dense::{
    cpu_session_stats, reset_cpu_session_stats, strided_batch_runs, CpuSessionStats,
    DefaultDenseExecutor, DenseError, DenseExecutor, DenseGemmBatchJob, DenseRead, DenseScalar,
    DenseView, DenseViewMut, DenseWrite, MatrixOp,
};

fn assert_close(got: f64, want: f64, tol: f64) {
    assert!((got - want).abs() <= tol, "{got} != {want} (tol {tol})");
}

// S1a session-scope evidence. `cpu_session_stats` is a process-wide counter,
// so every test that reads a delta takes this lock; they must not run
// concurrently with one another (other tests never touch the counter).
static CPU_SESSION_COUNTER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Held for the whole of each test below, not just around the measured call:
/// the reference loops open sessions too, and a concurrent test would show up
/// in another test's delta.
fn counter_lock() -> std::sync::MutexGuard<'static, ()> {
    CPU_SESSION_COUNTER_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn sessions_opened() -> u64 {
    cpu_session_stats().sessions_opened
}

// Heterogeneous jobs with a non-Identity operand op take the op-bearing serial
// route (`runs.len() == jobs.len()`). It must enter exactly one Tenferro
// session for the whole loop and land bitwise what the same jobs produce when
// issued one at a time through the public batch API.
#[allow(clippy::too_many_arguments)]
fn serial_route_is_one_session_and_bitwise_equal<T, W, R>(
    values: impl Fn(usize) -> T,
    accumulator: impl Fn(usize) -> T,
    alpha: DenseScalar,
    beta: DenseScalar,
    lhs_op: MatrixOp,
    rhs_op: MatrixOp,
    wrap_write: W,
    wrap_read: R,
) where
    T: Copy + PartialEq + std::fmt::Debug + 'static,
    W: for<'x> Fn(DenseViewMut<'x, T>) -> DenseWrite<'x>,
    R: for<'x> Fn(DenseView<'x, T>) -> DenseRead<'x>,
{
    let _guard = counter_lock();
    // Alternating shapes: no constant-stride run, so `strided_batch_runs`
    // yields singleton runs and the serial route is forced.
    let shapes = [(4usize, 3usize, 5usize), (5, 4, 3)];
    let mut jobs = Vec::new();
    let (mut lhs_offset, mut rhs_offset, mut dst_offset) = (0usize, 0usize, 0usize);
    for index in 0..8usize {
        let (rows, contracted, cols) = shapes[index % 2];
        jobs.push(DenseGemmBatchJob {
            rows,
            contracted,
            cols,
            lhs_offset,
            rhs_offset,
            dst_offset,
        });
        lhs_offset += rows * contracted + 3;
        rhs_offset += contracted * cols + 5;
        dst_offset += rows * cols + 7;
    }
    let runs = strided_batch_runs(&jobs);
    assert_eq!(
        runs,
        vec![1; jobs.len()],
        "fixture must force the serial route"
    );

    let lhs = (0..lhs_offset).map(&values).collect::<Vec<_>>();
    let rhs = (0..rhs_offset).map(|i| values(i + 11)).collect::<Vec<_>>();
    let initial = (0..dst_offset).map(&accumulator).collect::<Vec<_>>();
    let strides = [1usize];

    let mut batched = initial.clone();
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();
    let before = sessions_opened();
    let result = {
        executor.matmul_batch_axpby_with_ops_into(
            wrap_write(DenseViewMut::new(&mut batched, &[initial.len()], &strides, 0).unwrap()),
            wrap_read(DenseView::new(&lhs, &[lhs.len()], &strides, 0).unwrap()),
            wrap_read(DenseView::new(&rhs, &[rhs.len()], &strides, 0).unwrap()),
            &jobs,
            &runs,
            lhs_op,
            rhs_op,
            alpha,
            beta,
        )
    };
    let sessions = sessions_opened() - before;
    result.unwrap();
    assert_eq!(sessions, 1, "the whole job loop must share one session");

    // Reference: the identical kernels issued one job per call.
    let mut per_job = initial.clone();
    let mut reference = DefaultDenseExecutor::with_threads(1).unwrap();
    for job in &jobs {
        reference
            .matmul_batch_axpby_with_ops_into(
                wrap_write(DenseViewMut::new(&mut per_job, &[initial.len()], &strides, 0).unwrap()),
                wrap_read(DenseView::new(&lhs, &[lhs.len()], &strides, 0).unwrap()),
                wrap_read(DenseView::new(&rhs, &[rhs.len()], &strides, 0).unwrap()),
                std::slice::from_ref(job),
                &[1],
                lhs_op,
                rhs_op,
                alpha,
                beta,
            )
            .unwrap();
    }
    assert_eq!(batched, per_job, "session scope must not change any value");
    assert_ne!(batched, initial, "fixture must actually write");
}

#[test]
#[allow(clippy::redundant_closure)]
fn op_bearing_serial_batch_uses_one_session_f64() {
    serial_route_is_one_session_and_bitwise_equal(
        |i| 0.5 + 0.25 * i as f64,
        |i| 2.0 + 0.05 * i as f64,
        DenseScalar::F64(0.75),
        DenseScalar::F64(-0.25),
        MatrixOp::Transpose,
        MatrixOp::Identity,
        // Constructor functions fix one lifetime; the closures stay generic.
        |view| DenseWrite::F64(view),
        |view| DenseRead::F64(view),
    );
}

// The three operand-op pairs the tensor layer actually emits
// (`fusion_block.rs:1718-1719`, `:1742-1743`, `:1848-1852`); `Transpose` is
// never produced there.
#[test]
#[allow(clippy::redundant_closure)]
fn op_bearing_serial_batch_uses_one_session_c64() {
    for (lhs_op, rhs_op) in [
        (MatrixOp::Adjoint, MatrixOp::Identity),
        (MatrixOp::Identity, MatrixOp::Adjoint),
        (MatrixOp::Adjoint, MatrixOp::Adjoint),
    ] {
        serial_route_is_one_session_and_bitwise_equal(
            |i| Complex64::new(0.5 + 0.25 * i as f64, -0.75 + 0.125 * i as f64),
            |i| Complex64::new(2.0 + 0.05 * i as f64, -1.0 - 0.025 * i as f64),
            DenseScalar::C64(Complex64::new(0.75, -0.5)),
            DenseScalar::C64(Complex64::new(-0.25, 0.125)),
            lhs_op,
            rhs_op,
            // Constructor functions fix one lifetime; the closures stay generic.
            |view| DenseWrite::C64(view),
            |view| DenseRead::C64(view),
        );
    }
}

// An empty job list is a supported call on the op-bearing route
// (`runs.len() == jobs.len()` with both empty). A session is a process-wide
// critical section, so it must not be taken for an empty loop.
#[test]
fn op_bearing_empty_batch_opens_no_session() {
    let _guard = counter_lock();
    let strides = [1usize];
    let mut output = [3.0f64];
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();

    let before = sessions_opened();
    executor
        .matmul_batch_axpby_with_ops_into(
            DenseWrite::F64(DenseViewMut::new(&mut output, &[1], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&[1.0], &[1], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&[2.0], &[1], &strides, 0).unwrap()),
            &[],
            &[],
            MatrixOp::Transpose,
            MatrixOp::Identity,
            DenseScalar::F64(1.0),
            DenseScalar::F64(0.0),
        )
        .unwrap();
    assert_eq!(sessions_opened() - before, 0);
    assert_eq!(output, [3.0]);
}

// A failure raised inside the session must surface the same typed error the
// per-job path produced, and the executor must stay usable: the session is
// released on the error path, so a following call can enter a new one.
#[test]
fn op_bearing_serial_batch_error_inside_session_keeps_executor_usable() {
    let _guard = counter_lock();
    let good = DenseGemmBatchJob {
        rows: 2,
        contracted: 2,
        cols: 2,
        lhs_offset: 0,
        rhs_offset: 0,
        dst_offset: 0,
    };
    // Second job reads past the end of the operand buffers.
    let bad = DenseGemmBatchJob {
        rhs_offset: usize::MAX,
        ..good
    };
    let data = [1.0f64, 2.0, 3.0, 4.0];
    let mut output = [0.0f64; 4];
    let strides = [1usize];
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();

    let before = sessions_opened();
    let error = {
        executor
            .matmul_batch_axpby_with_ops_into(
                DenseWrite::F64(DenseViewMut::new(&mut output, &[4], &strides, 0).unwrap()),
                DenseRead::F64(DenseView::new(&data, &[4], &strides, 0).unwrap()),
                DenseRead::F64(DenseView::new(&data, &[4], &strides, 0).unwrap()),
                &[good, bad],
                &[1, 1],
                MatrixOp::Transpose,
                MatrixOp::Identity,
                DenseScalar::F64(1.0),
                DenseScalar::F64(0.0),
            )
            .unwrap_err()
    };
    let sessions = sessions_opened() - before;
    assert_eq!(error, DenseError::OutOfBounds);
    assert_eq!(sessions, 1);

    // The executor still works after the failed scope.
    let mut again = [0.0f64; 4];
    executor
        .matmul_batch_axpby_with_ops_into(
            DenseWrite::F64(DenseViewMut::new(&mut again, &[4], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&data, &[4], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&data, &[4], &strides, 0).unwrap()),
            &[good],
            &[1],
            MatrixOp::Transpose,
            MatrixOp::Identity,
            DenseScalar::F64(1.0),
            DenseScalar::F64(0.0),
        )
        .unwrap();
    // A^T A for the column-major [[1,3],[2,4]].
    assert_eq!(again, [5.0, 11.0, 11.0, 25.0]);
}

// `eig_vals` runs its contiguity pre-pass and the values call in one session
// (it opened two before), with the spectrum unchanged.
#[test]
fn eig_vals_uses_one_session_for_prepass_and_values() {
    let _guard = counter_lock();
    let shape = [2usize, 2];
    // Non-contiguous source, so the pre-pass actually copies.
    let padded = [2.0f64, 1.0, 9.0, 1.0, 3.0, 9.0];
    let strides = [1usize, 3];
    let mut executor = DefaultDenseExecutor::new();

    let before = sessions_opened();
    let values = {
        executor
            .eig_vals(DenseRead::F64(
                DenseView::new(&padded, &shape, &strides, 0).unwrap(),
            ))
            .unwrap()
    };
    let sessions = sessions_opened() - before;
    assert_eq!(sessions, 1);

    // [[2,1],[1,3]] is symmetric: the eigenvalues are real and known.
    let values = values.as_c64_slice().unwrap();
    assert_eq!(values.len(), 2);
    let mut real = values.iter().map(|v| v.re).collect::<Vec<_>>();
    real.sort_by(f64::total_cmp);
    let expected = [(5.0 - 5.0f64.sqrt()) / 2.0, (5.0 + 5.0f64.sqrt()) / 2.0];
    for (got, want) in real.iter().zip(expected) {
        assert_close(*got, want, 1e-12);
    }
    for value in values {
        assert_close(value.im, 0.0, 1e-12);
    }
}

#[test]
fn eig_vals_uses_one_session_for_c64() {
    let _guard = counter_lock();
    let c = Complex64::new;
    let shape = [2usize, 2];
    let strides = [1usize, 2];
    // [[0,-i],[i,0]] — Hermitian, eigenvalues -1 and +1.
    let m = [c(0.0, 0.0), c(0.0, 1.0), c(0.0, -1.0), c(0.0, 0.0)];
    let mut executor = DefaultDenseExecutor::new();

    let before = sessions_opened();
    let values = {
        executor
            .eig_vals(DenseRead::C64(
                DenseView::new(&m, &shape, &strides, 0).unwrap(),
            ))
            .unwrap()
    };
    let sessions = sessions_opened() - before;
    assert_eq!(sessions, 1);

    let values = values.as_c64_slice().unwrap();
    assert_eq!(values.len(), 2);
    let mut real = values.iter().map(|v| v.re).collect::<Vec<_>>();
    real.sort_by(f64::total_cmp);
    assert_close(real[0], -1.0, 1e-12);
    assert_close(real[1], 1.0, 1e-12);
}

// The counter is documented as a lower bound: `reset` zeroes it, and the
// sessions Tenferro opens internally for a plain backend-level dot are not
// visible here.
#[test]
fn cpu_session_stats_reset_and_scope() {
    let _guard = counter_lock();
    reset_cpu_session_stats();
    assert_eq!(cpu_session_stats(), CpuSessionStats::default());

    let data = [1.0f64, 2.0, 3.0, 4.0];
    let mut output = [0.0f64; 4];
    let strides = [1usize, 2];
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();
    executor
        .matmul_into(
            DenseWrite::F64(DenseViewMut::new(&mut output, &[2, 2], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&data, &[2, 2], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&data, &[2, 2], &strides, 0).unwrap()),
        )
        .unwrap();
    assert_eq!(cpu_session_stats().sessions_opened, 0);
    reset_cpu_session_stats();
}
