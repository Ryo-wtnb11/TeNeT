//! Cross-storage matrix for the typed methods unified over storage (#1756).
//!
//! Host rows always run; CUDA rows compare the device result with the Host
//! result on the same fixture and need a real device. Each #1756 leaf adds
//! its family.
//!
//! Oracles are hand sums over the entries the fixtures write
//! (`norm(t, p) = (Σ_c dim(c) Σ_ij |t_c[i,j]|^p)^(1/p)`, `norm(t, Inf) = max`,
//! `inner(a, b) = Σ_c dim(c) Σ_ij conj(a_c[i,j]) b_c[i,j]`), never read back
//! from a TeNeT result.
//!
//! Unsupported cells:
//! - CUDA `norm(p)` for `p != 2`, and `inner` with a lazy adjoint CUDA
//!   operand: `UnsupportedOnDevice`, asserted below.
//! - CUDA compact storage does not exist (`to_cuda` densifies), so the
//!   compact column is Host-only.
//! - Checked Generic and complex-coefficient rules have no CUDA methods
//!   (compile-time absent).

use std::sync::Arc;

use num_complex::Complex64;
use tenet::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use tenet::typed::{Error, GradedSpace, OperationError, Runtime, SectorSpectrum, TensorMap};

const POWERS: [f64; 4] = [1.0, 2.0, 3.0, f64::INFINITY];

fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= 1e-12 * (1.0 + expected.abs())
}

fn close_c(actual: Complex64, expected: Complex64) -> bool {
    (actual - expected).norm() <= 1e-12 * (1.0 + expected.norm())
}

/// `norm(t, p)` over `(dim(c), entry)` pairs.
fn norm_oracle(entries: &[(f64, Complex64)], p: f64) -> f64 {
    if p.is_infinite() {
        return entries.iter().map(|(_, x)| x.norm()).fold(0.0, f64::max);
    }
    entries
        .iter()
        .map(|(d, x)| d * x.norm().powf(p))
        .sum::<f64>()
        .powf(p.recip())
}

/// `inner(a, b)` over entries written in the same order.
fn inner_oracle(a: &[(f64, Complex64)], b: &[(f64, Complex64)]) -> Complex64 {
    a.iter()
        .zip(b)
        .map(|((d, x), (_, y))| *d * x.conj() * y)
        .sum()
}

fn entry(seed: f64, coupled: usize, i: usize, j: usize) -> Complex64 {
    Complex64::new(
        seed + 0.37 * i as f64 - 0.21 * j as f64 + 0.11 * coupled as f64,
        0.13 * seed - 0.17 * (i + 2 * j) as f64 + 0.05,
    )
}

/// Host rows for one rule and payload: dense, lazy adjoint and compact
/// operands of every reduction against the hand oracle, plus the admission
/// errors. The CUDA rows reuse the same fixtures through `$cuda`.
macro_rules! reduction_rows {
    ($runtime:expr, $leg:expr, $coupled:expr, $dim:expr, $D:ty, $conv:expr, $cuda:expr) => {{
        let runtime: &Runtime = $runtime;
        let leg = $leg;
        let coupled = $coupled;
        let dim = $dim;
        let conv = $conv;
        let build = |seed: f64| {
            let mut entries = Vec::new();
            let t = TensorMap::<_, $D>::from_subblock_fn(runtime, [&leg], [&leg], |trees, ix| {
                let c = coupled(trees.coupled());
                let x = conv(entry(seed, c, ix[0], ix[1]));
                entries.push((dim(c), x));
                x
            })
            .unwrap();
            let wide: Vec<(f64, Complex64)> = entries
                .into_iter()
                .map(|(d, x): (f64, $D)| (d, Complex64::from(x)))
                .collect();
            (t, wide)
        };
        let (a, a_entries) = build(0.9);
        let (b, b_entries) = build(-0.4);

        // Dense and lazy adjoint: every entrywise norm is adjoint invariant.
        let lazy_a = a.adjoint().unwrap();
        let lazy_b = b.adjoint().unwrap();
        for p in POWERS {
            let expected = norm_oracle(&a_entries, p);
            assert!(close(a.norm(p).unwrap(), expected), "dense norm({p})");
            assert!(close(lazy_a.norm(p).unwrap(), expected), "lazy norm({p})");
        }
        let ab = inner_oracle(&a_entries, &b_entries);
        assert!(close_c(Complex64::from(a.inner(&b).unwrap()), ab));
        // `<a^H, b^H> = conj(<a, b>)`, with dim(c) = dim(conj(c)).
        assert!(close_c(Complex64::from(lazy_a.inner(&lazy_b).unwrap()), ab.conj()));
        let materialized = lazy_a.materialize().unwrap();
        assert!(close_c(
            Complex64::from(lazy_a.inner(&materialized).unwrap()),
            Complex64::from(materialized.inner(&materialized).unwrap()),
        ));

        // Compact diagonal: the stored diagonal only.
        let mut diag_entries = Vec::new();
        let spectra: Vec<_> = leg
            .sectors()
            .unwrap()
            .into_iter()
            .map(|sector| {
                let k = leg.degeneracy(&sector).unwrap();
                let c = coupled(&sector);
                let values: Vec<$D> = (0..k).map(|i| conv(entry(0.6, c, i, i))).collect();
                diag_entries.extend(values.iter().map(|&x| (dim(c), Complex64::from(x))));
                SectorSpectrum { sector, values }
            })
            .collect();
        let compact = TensorMap::<_, $D>::diagonal(runtime, &leg, spectra).unwrap();
        for p in POWERS {
            let expected = norm_oracle(&diag_entries, p);
            assert!(close(compact.norm(p).unwrap(), expected), "compact norm({p})");
        }
        assert!(close_c(
            Complex64::from(compact.inner(&compact).unwrap()),
            inner_oracle(&diag_entries, &diag_entries),
        ));

        // Admission: invalid `p` and a foreign space.
        assert!(matches!(a.norm(0.0), Err(Error::InvalidArgument(_))));
        let wider = TensorMap::<_, $D>::from_subblock_fn(runtime, [&leg, &leg], [&leg], |_, _| {
            conv(Complex64::new(1.0, 0.0))
        })
        .unwrap();
        assert!(matches!(
            a.inner(&wider),
            Err(Error::Operation(op)) if matches!(*op, OperationError::SpaceMismatch { .. })
        ));

        $cuda(&a, &b, &lazy_a);
    }};
}

fn no_cuda<T>(_: &T, _: &T, _: &T) {}

fn u1_leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 3),
            (U1Irrep::new(-1), 1),
        ],
    )
    .unwrap()
}

fn su2_leg() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 3),
            (SU2Irrep::from_twice_spin(2), 1),
        ],
    )
    .unwrap()
}

fn u1_coupled(c: &U1Irrep) -> usize {
    c.charge().unsigned_abs() as usize
}

fn su2_coupled(c: &SU2Irrep) -> usize {
    c.twice_spin()
}

#[test]
fn host_reductions_match_the_hand_oracle() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let real = |x: Complex64| x.re;
    let complex = |x: Complex64| x;
    let one = |_: usize| 1.0;
    let su2_dim = |twice: usize| (twice + 1) as f64;
    reduction_rows!(&runtime, u1_leg(), u1_coupled, one, f64, real, no_cuda);
    reduction_rows!(
        &runtime,
        u1_leg(),
        u1_coupled,
        one,
        Complex64,
        complex,
        no_cuda
    );
    reduction_rows!(
        &runtime,
        su2_leg(),
        su2_coupled,
        su2_dim,
        f64,
        real,
        no_cuda
    );
    reduction_rows!(
        &runtime,
        su2_leg(),
        su2_coupled,
        su2_dim,
        Complex64,
        complex,
        no_cuda
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn cuda_reductions_match_host() {
    use tenet::sector::{CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec};
    use tenet::typed::CudaPayload;

    fn device_rows<R, D>(a: &TensorMap<R, D>, b: &TensorMap<R, D>, lazy_a: &TensorMap<R, D>)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: CudaPayload + Into<Complex64> + std::fmt::Debug,
    {
        let (a_dev, b_dev) = (a.to_cuda().unwrap(), b.to_cuda().unwrap());
        let host = a.norm(2.0).unwrap();
        assert!(close(a_dev.norm(2.0).unwrap(), host));
        assert!(close(lazy_a.to_cuda().unwrap().norm(2.0).unwrap(), host));
        assert!(close(a_dev.adjoint().unwrap().norm(2.0).unwrap(), host));
        assert!(close_c(
            a_dev.inner(&b_dev).unwrap().into(),
            a.inner(b).unwrap().into(),
        ));
        assert_eq!(
            a_dev.norm(3.0).unwrap_err(),
            Error::UnsupportedOnDevice("device norm supports only p = 2, got 3".into())
        );
        assert_eq!(
            a_dev.norm(f64::INFINITY).unwrap_err(),
            Error::UnsupportedOnDevice("device norm supports only p = 2, got inf".into())
        );
        assert_eq!(
            a_dev.inner(&a_dev.adjoint().unwrap()).unwrap_err(),
            Error::UnsupportedOnDevice("inner does not support lazy adjoint CUDA operands".into())
        );
        assert!(matches!(a_dev.norm(-1.0), Err(Error::InvalidArgument(_))));
    }

    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let real = |x: Complex64| x.re;
    let complex = |x: Complex64| x;
    let one = |_: usize| 1.0;
    let su2_dim = |twice: usize| (twice + 1) as f64;
    reduction_rows!(&runtime, u1_leg(), u1_coupled, one, f64, real, device_rows);
    reduction_rows!(
        &runtime,
        u1_leg(),
        u1_coupled,
        one,
        Complex64,
        complex,
        device_rows
    );
    reduction_rows!(
        &runtime,
        su2_leg(),
        su2_coupled,
        su2_dim,
        f64,
        real,
        device_rows
    );
    reduction_rows!(
        &runtime,
        su2_leg(),
        su2_coupled,
        su2_dim,
        Complex64,
        complex,
        device_rows
    );
}
