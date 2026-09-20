//! Adapter-level device tests over every admitted [`CudaScalar`] payload
//! (`f32`, `f64`, `Complex32`, `Complex64`), issue #1326 (leaf C1).
//!
//! What this file owns that the `f64`/`Complex64` suites do not: the paths
//! that were typed by `f64` rather than by the payload's real lane — the
//! spectrum and reduction downloads, the rank-0 divisor of the Hermitian
//! test, and its `64 * eps` tolerance — plus the one dtype the device cannot
//! serve (complex device QR, tenferro-rs#1833 and #1271).
//!
//! Oracles are host computations in **double precision** over the widened
//! fixture: a single-precision device answer is compared against `f64` host
//! arithmetic on the same values, never against a second single-precision run
//! and never against a TeNeT descriptor. Tolerances are written in epsilons of
//! the payload's own lane, so no assertion carries a platform-dependent
//! absolute constant.
//!
//! Run with `cargo test -p tenet-dense --no-default-features --features \
//! cuda,cpu-faer --test cuda_scalar_dtypes -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use std::fmt::Debug;

use num_complex::{Complex32, Complex64};
use tenet_dense::{
    cuda_eigh_region, cuda_gemm_region_with_ops_into, cuda_is_hermitian_region, cuda_qr_region,
    cuda_svd_region, cuda_transfer_stats, CudaDenseContext, CudaDenseStorage, CudaScalar,
    DenseDType, DenseError, MatrixOp,
};

/// The payload dtypes under test, with just enough host arithmetic for a
/// double-precision oracle over the widened fixture.
trait ProbeScalar: CudaScalar + Copy + Debug + PartialEq {
    const NAME: &'static str;
    /// Machine epsilon of this payload's real lane, widened.
    const EPSILON: f64;

    /// The payload value nearest `value`, i.e. the fixture as the device sees
    /// it. The oracle runs on `widen(narrow(z))`, so the comparison isolates
    /// the device's error rather than the fixture's rounding.
    fn narrow(value: Complex64) -> Self;
    fn widen(self) -> Complex64;
}

impl ProbeScalar for f32 {
    const NAME: &'static str = "f32";
    const EPSILON: f64 = f32::EPSILON as f64;

    fn narrow(value: Complex64) -> Self {
        value.re as Self
    }

    fn widen(self) -> Complex64 {
        Complex64::new(f64::from(self), 0.0)
    }
}

impl ProbeScalar for f64 {
    const NAME: &'static str = "f64";
    const EPSILON: f64 = f64::EPSILON;

    fn narrow(value: Complex64) -> Self {
        value.re
    }

    fn widen(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
}

impl ProbeScalar for Complex32 {
    const NAME: &'static str = "Complex32";
    const EPSILON: f64 = f32::EPSILON as f64;

    fn narrow(value: Complex64) -> Self {
        Complex32::new(value.re as f32, value.im as f32)
    }

    fn widen(self) -> Complex64 {
        Complex64::new(f64::from(self.re), f64::from(self.im))
    }
}

impl ProbeScalar for Complex64 {
    const NAME: &'static str = "Complex64";
    const EPSILON: f64 = f64::EPSILON;

    fn narrow(value: Complex64) -> Self {
        value
    }

    fn widen(self) -> Complex64 {
        self
    }
}

fn context() -> CudaDenseContext {
    CudaDenseContext::new(0).expect("CUDA device 0 must be available for the device suite")
}

fn upload<D: ProbeScalar>(ctx: &CudaDenseContext, data: &[D]) -> CudaDenseStorage {
    CudaDenseStorage::upload::<D>(ctx, data).expect("upload")
}

fn download<D: ProbeScalar>(ctx: &CudaDenseContext, storage: &CudaDenseStorage) -> Vec<D> {
    storage.download::<D>(ctx).expect("download")
}

/// A deterministic, non-degenerate `rows x cols` column-major fixture, as the
/// payload sees it and as the oracle sees it.
fn fixture<D: ProbeScalar>(rows: usize, cols: usize, seed: f64) -> (Vec<D>, Vec<Complex64>) {
    let payload: Vec<D> = (0..rows * cols)
        .map(|index| {
            let index = index as f64;
            D::narrow(Complex64::new(
                seed + 0.5 + index * 0.25 - (index * index) * 0.03125,
                -0.375 + index * 0.125,
            ))
        })
        .collect();
    let wide = payload.iter().map(|value| value.widen()).collect();
    (payload, wide)
}

/// `k * sqrt(n) * eps(real(D))`, relative to the magnitude being compared:
/// the error a chain of `n` accumulated products of this lane is entitled to.
fn tolerance<D: ProbeScalar>(terms: usize, scale: f64) -> f64 {
    16.0 * (terms as f64).sqrt() * D::EPSILON * scale.max(1.0)
}

fn assert_close<D: ProbeScalar>(actual: &[D], expected: &[Complex64], terms: usize, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        let scale = want.norm();
        let allowed = tolerance::<D>(terms, scale);
        assert!(
            (got.widen() - want).norm() <= allowed,
            "{what} ({}): element {index} got {got:?} want {want} (allowed {allowed:e})",
            D::NAME
        );
    }
}

// ---------------------------------------------------------------------------
// GEMM regions: every operand-op pair, in every payload dtype.
// ---------------------------------------------------------------------------

/// The logical `m x k` operand a [`MatrixOp`] reads out of a packed parent.
fn operand(op: MatrixOp, parent: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
    // `Identity` reads the parent as `rows x cols`; the other two read it as
    // `cols x rows` and transpose (and conjugate) it into `rows x cols`.
    (0..rows * cols)
        .map(|index| {
            let (row, col) = (index % rows, index / rows);
            match op {
                MatrixOp::Identity => parent[row + col * rows],
                MatrixOp::Transpose => parent[col + row * cols],
                MatrixOp::Adjoint => parent[col + row * cols].conj(),
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn gemm_case<D: ProbeScalar>(
    ctx: &mut CudaDenseContext,
    lhs_op: MatrixOp,
    rhs_op: MatrixOp,
    m: usize,
    k: usize,
    n: usize,
    alpha: Complex64,
    beta: Complex64,
) {
    let (lhs_payload, lhs_wide) = fixture::<D>(m, k, 0.0);
    let (rhs_payload, rhs_wide) = fixture::<D>(k, n, 1.25);
    let (dst_payload, dst_wide) = fixture::<D>(m, n, -2.0);

    let lhs = upload::<D>(ctx, &lhs_payload);
    let rhs = upload::<D>(ctx, &rhs_payload);
    let mut dst = upload::<D>(ctx, &dst_payload);

    cuda_gemm_region_with_ops_into::<D>(
        ctx,
        &mut dst,
        0,
        &lhs,
        0,
        &rhs,
        0,
        m,
        k,
        n,
        lhs_op,
        rhs_op,
        D::narrow(alpha),
        D::narrow(beta),
    )
    .unwrap_or_else(|err| panic!("{} GEMM {lhs_op:?}/{rhs_op:?} failed: {err}", D::NAME));

    // Double-precision oracle over the widened operands, with the payload's
    // own rounding of alpha/beta so only the device's error is left.
    let alpha = D::narrow(alpha).widen();
    let beta = D::narrow(beta).widen();
    let lhs_view = operand(lhs_op, &lhs_wide, m, k);
    let rhs_view = operand(rhs_op, &rhs_wide, k, n);
    let expected: Vec<Complex64> = (0..m * n)
        .map(|index| {
            let (row, col) = (index % m, index / m);
            let mut acc = Complex64::new(0.0, 0.0);
            for step in 0..k {
                acc += lhs_view[row + step * m] * rhs_view[step + col * k];
            }
            alpha * acc + beta * dst_wide[index]
        })
        .collect();

    assert_close::<D>(
        &download::<D>(ctx, &dst),
        &expected,
        k + 1,
        &format!("GEMM {lhs_op:?}/{rhs_op:?} {m}x{k}x{n}"),
    );
}

fn gemm_sweep<D: ProbeScalar>(ctx: &mut CudaDenseContext) {
    let ops = [MatrixOp::Identity, MatrixOp::Transpose, MatrixOp::Adjoint];
    for lhs_op in ops {
        for rhs_op in ops {
            for &(m, k, n) in &[(3usize, 4usize, 2usize), (1, 6, 1), (4, 4, 4)] {
                gemm_case::<D>(
                    ctx,
                    lhs_op,
                    rhs_op,
                    m,
                    k,
                    n,
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                );
                gemm_case::<D>(
                    ctx,
                    lhs_op,
                    rhs_op,
                    m,
                    k,
                    n,
                    Complex64::new(-0.75, 0.5),
                    Complex64::new(1.0, 0.0),
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn region_gemm_matches_a_double_precision_oracle_for_every_dtype_and_op_pair() {
    let mut ctx = context();
    gemm_sweep::<f32>(&mut ctx);
    gemm_sweep::<f64>(&mut ctx);
    gemm_sweep::<Complex32>(&mut ctx);
    gemm_sweep::<Complex64>(&mut ctx);
}

// ---------------------------------------------------------------------------
// Factorization regions: the laws, and the widened spectra.
// ---------------------------------------------------------------------------

/// `U diag(s) Vt == A`, `U^H U == I`, `Vt Vt^H == I`, and a spectrum that is
/// non-negative, descending and downloaded as `f64` whatever the lane.
fn svd_case<D: ProbeScalar>(ctx: &mut CudaDenseContext, rows: usize, cols: usize) {
    let k = rows.min(cols);
    let (payload, wide) = fixture::<D>(rows, cols, 0.0);
    let src = upload::<D>(ctx, &payload);

    let (u, values, vt) = cuda_svd_region::<D>(ctx, &src, 0, rows, cols)
        .unwrap_or_else(|err| panic!("{} SVD {rows}x{cols}: {err}", D::NAME));
    let u: Vec<Complex64> = download::<D>(ctx, &u).iter().map(|v| v.widen()).collect();
    let vt: Vec<Complex64> = download::<D>(ctx, &vt).iter().map(|v| v.widen()).collect();

    assert_eq!(values.len(), k);
    for window in values.windows(2) {
        assert!(
            window[0] >= window[1] && window[1] >= 0.0,
            "{}: singular values must be non-negative and descending: {values:?}",
            D::NAME
        );
    }

    let scale = values[0].max(1.0);
    let allowed = tolerance::<D>(rows * cols, scale);
    for col in 0..cols {
        for row in 0..rows {
            let mut acc = Complex64::new(0.0, 0.0);
            for index in 0..k {
                acc += u[row + index * rows] * values[index] * vt[index + col * k];
            }
            let residual = (acc - wide[row + col * rows]).norm();
            assert!(
                residual <= allowed,
                "{}: SVD reconstruction at ({row}, {col}) off by {residual:e} (allowed {allowed:e})",
                D::NAME
            );
        }
    }
    assert_orthonormal_columns::<D>(&u, rows, k, "SVD U");
    assert_orthonormal_rows::<D>(&vt, k, cols, "SVD Vt");
}

/// `Q R == A`, `Q^H Q == I`, and the positive-diagonal gauge: `R_jj` real and
/// non-negative.
fn qr_case<D: ProbeScalar>(ctx: &mut CudaDenseContext, rows: usize, cols: usize) {
    let k = rows.min(cols);
    let (payload, wide) = fixture::<D>(rows, cols, 0.0);
    let src = upload::<D>(ctx, &payload);

    let (q, r) = cuda_qr_region::<D>(ctx, &src, 0, rows, cols)
        .unwrap_or_else(|err| panic!("{} QR {rows}x{cols}: {err}", D::NAME));
    let q: Vec<Complex64> = download::<D>(ctx, &q).iter().map(|v| v.widen()).collect();
    let r: Vec<Complex64> = download::<D>(ctx, &r).iter().map(|v| v.widen()).collect();

    let scale = wide
        .iter()
        .fold(1.0_f64, |acc, value| acc.max(value.norm()));
    let allowed = tolerance::<D>(rows * cols, scale);
    for col in 0..cols {
        for row in 0..rows {
            let mut acc = Complex64::new(0.0, 0.0);
            for index in 0..k {
                acc += q[row + index * rows] * r[index + col * k];
            }
            let residual = (acc - wide[row + col * rows]).norm();
            assert!(
                residual <= allowed,
                "{}: QR reconstruction at ({row}, {col}) off by {residual:e}",
                D::NAME
            );
        }
    }
    assert_orthonormal_columns::<D>(&q, rows, k, "QR Q");
    for index in 0..k {
        let diagonal = r[index + index * k];
        assert!(
            diagonal.re >= -allowed && diagonal.im.abs() <= allowed,
            "{}: positive-diagonal gauge broken at {index}: {diagonal}",
            D::NAME
        );
    }
}

/// `A v_j == lambda_j v_j`, eigenvalues real and ascending, vectors
/// orthonormal.
fn eigh_case<D: ProbeScalar>(ctx: &mut CudaDenseContext, n: usize) {
    let (payload, _) = fixture::<D>(n, n, 0.0);
    // Hermitian by construction: `(A + A^H) / 2`, formed in the payload dtype
    // so the device sees an exactly Hermitian block.
    let hermitian: Vec<D> = (0..n * n)
        .map(|index| {
            let (row, col) = (index % n, index / n);
            let upper = payload[row + col * n].widen();
            let lower = payload[col + row * n].widen();
            D::narrow((upper + lower.conj()) * 0.5)
        })
        .collect();
    let wide: Vec<Complex64> = hermitian.iter().map(|value| value.widen()).collect();
    let src = upload::<D>(ctx, &hermitian);

    assert!(
        cuda_is_hermitian_region::<D>(ctx, &src, 0, n)
            .unwrap_or_else(|err| panic!("{}: hermitian test: {err}", D::NAME)),
        "{}: an exactly Hermitian block must be admitted",
        D::NAME
    );

    let (values, vectors) = cuda_eigh_region::<D>(ctx, &src, 0, n)
        .unwrap_or_else(|err| panic!("{} EIGH {n}x{n}: {err}", D::NAME));
    let vectors: Vec<Complex64> = download::<D>(ctx, &vectors)
        .iter()
        .map(|v| v.widen())
        .collect();

    assert_eq!(values.len(), n);
    for window in values.windows(2) {
        assert!(
            window[0] <= window[1],
            "{}: eigenvalues must ascend: {values:?}",
            D::NAME
        );
    }

    let scale = values
        .iter()
        .fold(1.0_f64, |acc, value| acc.max(value.abs()));
    let allowed = tolerance::<D>(n * n, scale);
    for (index, &value) in values.iter().enumerate() {
        for row in 0..n {
            let mut acc = Complex64::new(0.0, 0.0);
            for step in 0..n {
                acc += wide[row + step * n] * vectors[step + index * n];
            }
            let residual = (acc - vectors[row + index * n] * value).norm();
            assert!(
                residual <= allowed,
                "{}: eigenpair {index} residual at row {row} is {residual:e}",
                D::NAME
            );
        }
    }
    assert_orthonormal_columns::<D>(&vectors, n, n, "EIGH vectors");
}

fn assert_orthonormal_columns<D: ProbeScalar>(
    matrix: &[Complex64],
    rows: usize,
    cols: usize,
    what: &str,
) {
    let allowed = tolerance::<D>(rows, 1.0);
    for left in 0..cols {
        for right in 0..cols {
            let mut acc = Complex64::new(0.0, 0.0);
            for row in 0..rows {
                acc += matrix[row + left * rows].conj() * matrix[row + right * rows];
            }
            let want = Complex64::new(f64::from(u8::from(left == right)), 0.0);
            assert!(
                (acc - want).norm() <= allowed,
                "{what} ({}): columns {left},{right} give {acc} (allowed {allowed:e})",
                D::NAME
            );
        }
    }
}

fn assert_orthonormal_rows<D: ProbeScalar>(
    matrix: &[Complex64],
    rows: usize,
    cols: usize,
    what: &str,
) {
    let allowed = tolerance::<D>(cols, 1.0);
    for left in 0..rows {
        for right in 0..rows {
            let mut acc = Complex64::new(0.0, 0.0);
            for col in 0..cols {
                acc += matrix[left + col * rows] * matrix[right + col * rows].conj();
            }
            let want = Complex64::new(f64::from(u8::from(left == right)), 0.0);
            assert!(
                (acc - want).norm() <= allowed,
                "{what} ({}): rows {left},{right} give {acc} (allowed {allowed:e})",
                D::NAME
            );
        }
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn region_svd_obeys_its_laws_for_every_dtype() {
    let mut ctx = context();
    for &(rows, cols) in &[(5usize, 3usize), (3, 5), (4, 4)] {
        svd_case::<f32>(&mut ctx, rows, cols);
        svd_case::<f64>(&mut ctx, rows, cols);
        svd_case::<Complex32>(&mut ctx, rows, cols);
        svd_case::<Complex64>(&mut ctx, rows, cols);
    }
}

/// Only the real payloads: both complex dtypes fail Tenferro 0.5.0's
/// positive-diagonal `triu` kernel and are rejected at the boundary instead
/// (`complex_device_qr_is_rejected_before_any_device_work`).
#[test]
#[ignore = "requires a real CUDA device"]
fn region_qr_obeys_its_laws_for_every_supported_dtype() {
    let mut ctx = context();
    for &(rows, cols) in &[(5usize, 3usize), (4, 4)] {
        qr_case::<f32>(&mut ctx, rows, cols);
        qr_case::<f64>(&mut ctx, rows, cols);
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn region_eigh_obeys_its_laws_for_every_dtype() {
    let mut ctx = context();
    for n in [4usize, 6] {
        eigh_case::<f32>(&mut ctx, n);
        eigh_case::<f64>(&mut ctx, n);
        eigh_case::<Complex32>(&mut ctx, n);
        eigh_case::<Complex64>(&mut ctx, n);
    }
}

// ---------------------------------------------------------------------------
// The real lane itself: spectra widened to f64, and the rank-0 divisor typed
// by the lane rather than by f64 or by the payload.
// ---------------------------------------------------------------------------

/// A device spectrum of a single-precision payload comes back as `F32` and is
/// widened here, so the `f64` values TeNeT's truncation decisions consume
/// exist for every dtype. The oracle is the known spectrum of a diagonal
/// matrix, which no factorization has to be trusted for.
fn spectrum_widening_case<D: ProbeScalar>(ctx: &mut CudaDenseContext) {
    let n = 4usize;
    let diagonal = [4.0_f64, 3.0, 2.0, 0.5];
    let payload: Vec<D> = (0..n * n)
        .map(|index| {
            let (row, col) = (index % n, index / n);
            D::narrow(Complex64::new(
                if row == col { diagonal[row] } else { 0.0 },
                0.0,
            ))
        })
        .collect();
    let src = upload::<D>(ctx, &payload);

    let (_, values, _) = cuda_svd_region::<D>(ctx, &src, 0, n, n).expect("svd");
    let allowed = tolerance::<D>(n, 4.0);
    for (index, value) in values.iter().enumerate() {
        assert!(
            (value - diagonal[index]).abs() <= allowed,
            "{}: singular value {index} is {value}, expected {}",
            D::NAME,
            diagonal[index]
        );
    }

    let (eigenvalues, _) = cuda_eigh_region::<D>(ctx, &src, 0, n).expect("eigh");
    let mut ascending = diagonal;
    ascending.sort_by(f64::total_cmp);
    for (index, value) in eigenvalues.iter().enumerate() {
        assert!(
            (value - ascending[index]).abs() <= allowed,
            "{}: eigenvalue {index} is {value}, expected {}",
            D::NAME,
            ascending[index]
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_spectra_are_widened_to_f64_for_every_dtype() {
    let mut ctx = context();
    spectrum_widening_case::<f32>(&mut ctx);
    spectrum_widening_case::<f64>(&mut ctx);
    spectrum_widening_case::<Complex32>(&mut ctx);
    spectrum_widening_case::<Complex64>(&mut ctx);
}

/// The Hermitian test divides the block by a rank-0 **real** scalar of the
/// payload's lane. An `f64` divisor against an `F32`/`C32` block and a
/// payload-typed complex divisor are both rejected by Tenferro, so a block
/// this test admits is evidence that the divisor is typed by the lane: the
/// whole chain would error out otherwise, not merely answer differently.
fn hermitian_admits_an_exactly_hermitian_block<D: ProbeScalar>(ctx: &mut CudaDenseContext) {
    let n = 3usize;
    let payload: Vec<D> = (0..n * n)
        .map(|index| {
            let (row, col) = (index % n, index / n);
            let (low, high) = (row.min(col), row.max(col));
            let magnitude = 1.0 + (low as f64) + 0.25 * (high as f64);
            let imaginary = if row == col {
                0.0
            } else if row < col {
                0.5 + low as f64
            } else {
                -(0.5 + low as f64)
            };
            D::narrow(Complex64::new(magnitude, imaginary))
        })
        .collect();
    let src = upload::<D>(ctx, &payload);
    assert!(
        cuda_is_hermitian_region::<D>(ctx, &src, 0, n).expect("hermitian"),
        "{}: an exactly Hermitian block must be admitted",
        D::NAME
    );
}

/// The C1 tolerance change, on device: a block whose anti-Hermitian residual
/// is a few epsilons **of the payload's own lane** is Hermitian, and one whose
/// residual is far outside that lane is not. For `f32` the accepted block is
/// the one the old `64 * eps(f64)` constant rejected; for `f64` both decisions
/// are what they always were.
fn hermitian_tolerance_case<D: ProbeScalar>(ctx: &mut CudaDenseContext) {
    // [[1, delta], [0, 1]]: the shared half-residual rule changes truth value
    // at `delta = 128 * eps` up to a negligible O(delta^2).
    let block = |ctx: &CudaDenseContext, epsilons: f64| {
        let delta = epsilons * D::EPSILON;
        upload::<D>(
            ctx,
            &[
                D::narrow(Complex64::new(1.0, 0.0)),
                D::narrow(Complex64::new(0.0, 0.0)),
                D::narrow(Complex64::new(delta, 0.0)),
                D::narrow(Complex64::new(1.0, 0.0)),
            ],
        )
    };

    let below = block(ctx, 120.0);
    assert!(
        cuda_is_hermitian_region::<D>(ctx, &below, 0, 2).expect("below"),
        "{}: a residual of 120 eps(real(D)) must be admitted",
        D::NAME
    );

    let above = block(ctx, 4096.0);
    assert!(
        !cuda_is_hermitian_region::<D>(ctx, &above, 0, 2).expect("above"),
        "{}: a residual of 4096 eps(real(D)) must be rejected",
        D::NAME
    );

    // A block that is non-Hermitian by a whole unit is rejected in any lane.
    let asymmetric = upload::<D>(
        ctx,
        &[
            D::narrow(Complex64::new(1.0, 0.0)),
            D::narrow(Complex64::new(0.0, 0.0)),
            D::narrow(Complex64::new(1.0, 0.0)),
            D::narrow(Complex64::new(1.0, 0.0)),
        ],
    );
    assert!(
        !cuda_is_hermitian_region::<D>(ctx, &asymmetric, 0, 2).expect("asymmetric"),
        "{}: a clearly non-Hermitian block must be rejected",
        D::NAME
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn the_hermitian_rule_scales_with_the_payload_real_lane() {
    let mut ctx = context();
    hermitian_admits_an_exactly_hermitian_block::<f32>(&mut ctx);
    hermitian_admits_an_exactly_hermitian_block::<f64>(&mut ctx);
    hermitian_admits_an_exactly_hermitian_block::<Complex32>(&mut ctx);
    hermitian_admits_an_exactly_hermitian_block::<Complex64>(&mut ctx);

    hermitian_tolerance_case::<f32>(&mut ctx);
    hermitian_tolerance_case::<f64>(&mut ctx);
    hermitian_tolerance_case::<Complex32>(&mut ctx);
    hermitian_tolerance_case::<Complex64>(&mut ctx);

    // The hazard itself: at 120 single-precision epsilons the `f32` block is
    // Hermitian, while the same *relative* residual in `f64` — which is what
    // the pre-C1 constant measured every dtype against — is not.
    let single = 120.0 * f32::EPSILON;
    let double_lane_residual = f64::from(single) / f64::EPSILON;
    assert!(
        double_lane_residual > 128.0,
        "the f32 fixture must be far outside the f64 threshold, not merely near it"
    );
}

// ---------------------------------------------------------------------------
// Capability boundary: Complex32 device QR.
// ---------------------------------------------------------------------------

/// The positive-diagonal gauge's `triu` kernel materializes a complex zero,
/// which the pinned Tenferro's NVRTC cannot construct for `cuFloatComplex`
/// (tenferro-rs#1833) or `cuDoubleComplex` (#1271). The adapter rejects both
/// dtypes *before* any device work, which is observable as untouched
/// counters — the failure would otherwise arrive as an NVRTC compile log
/// wrapped in a backend error, after a submission.
fn qr_rejection_case<D: ProbeScalar>(ctx: &mut CudaDenseContext) {
    let (payload, _) = fixture::<D>(4, 3, 0.0);
    let src = upload::<D>(ctx, &payload);

    let before = cuda_transfer_stats();
    let Err(err) = cuda_qr_region::<D>(ctx, &src, 0, 4, 3) else {
        panic!("{} device QR must be unsupported", D::NAME);
    };
    let after = cuda_transfer_stats();

    assert!(
        matches!(err, DenseError::Unsupported { op: "cuda_qr", .. }),
        "{}: expected a typed capability error, got {err}",
        D::NAME
    );
    assert!(err.to_string().contains("1833"), "{}: {err}", D::NAME);
    assert_eq!(
        after.solver_calls,
        before.solver_calls,
        "{}: no cuSOLVER submission may happen",
        D::NAME
    );
    assert_eq!(after.gemm_calls, before.gemm_calls);
    assert_eq!(after.h2d_calls, before.h2d_calls);
    assert_eq!(after.d2h_calls, before.d2h_calls);
    assert_eq!(after.device_allocs, before.device_allocs);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn complex_device_qr_is_rejected_before_any_device_work() {
    let mut ctx = context();
    qr_rejection_case::<Complex32>(&mut ctx);
    qr_rejection_case::<Complex64>(&mut ctx);

    // `f32` QR is supported: the boundary is complexity, not single precision.
    let (single, _) = fixture::<f32>(4, 3, 0.0);
    let single = upload::<f32>(&ctx, &single);
    assert!(cuda_qr_region::<f32>(&mut ctx, &single, 0, 4, 3).is_ok());
}

/// A dtype mismatch stays a typed error for the new payloads too, rather than
/// a reinterpretation of the bytes: `f32` and `Complex32` buffers are half the
/// width of their double-precision twins.
#[test]
#[ignore = "requires a real CUDA device"]
fn every_admitted_dtype_reports_its_own_tag_and_rejects_the_others() {
    let ctx = context();
    let single = upload::<f32>(&ctx, &[1.0f32, 2.0, 3.0, 4.0]);
    let single_complex = upload::<Complex32>(&ctx, &[Complex32::new(1.0, -1.0); 4]);

    assert_eq!(single.dtype(), DenseDType::F32);
    assert_eq!(single_complex.dtype(), DenseDType::C32);
    assert!(matches!(
        single.download::<f64>(&ctx),
        Err(DenseError::DTypeMismatch {
            expected: DenseDType::F64,
            actual: DenseDType::F32,
            ..
        })
    ));
    assert!(matches!(
        single_complex.download::<Complex64>(&ctx),
        Err(DenseError::DTypeMismatch {
            expected: DenseDType::C64,
            actual: DenseDType::C32,
            ..
        })
    ));
}
