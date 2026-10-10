//! Cross-storage matrix for the typed methods unified over storage (#1756).
//!
//! Host rows always run; CUDA rows compare the device result with the Host
//! result on the same fixture and need a real device. Each #1756 leaf adds
//! its family.
//!
//! Oracles are hand sums over the entries the fixtures write
//! (`norm(t, p) = (Σ_c dim(c) Σ_ij |t_c[i,j]|^p)^(1/p)`, `norm(t, Inf) = max`,
//! `inner(a, b) = Σ_c dim(c) Σ_ij conj(a_c[i,j]) b_c[i,j]`), never read back
//! from a TeNeT result. Elementwise results (`scale`, `zeros_like`, `axpby`,
//! `axpby_into`) are checked through `norm` and the linear functional
//! `inner(c, ·)` against the entrywise combination of the written entries;
//! fixtures on one space are written in one order, so their entries align.
//! Compact factors (`svd_compact`, `qr_compact`) are checked the same way
//! through their rebuilt product, and through the norm of `s`/`r`, which an
//! isometric outer factor preserves. `eigh_full` runs on the Hermitian
//! fixture `h_c[i,j] = e_c[i,j] + conj(e_c[j,i])` and is checked through
//! `v d v^H` and the norm of `d`. `twist` runs on fermion parity (the U(1)
//! and SU(2) fixtures are bosonic, where it is the identity) and is checked
//! through `inner(t, twist(t))` against `Σ θ |t|²`, with θ = −1 for each
//! listed leg whose uncoupled sector is odd.
//!
//! Unsupported cells:
//! - CUDA `norm(p)` for `p != 2`, and `inner` with a lazy adjoint CUDA
//!   operand: `UnsupportedOnDevice`, asserted below.
//! - CUDA `axpby` of one lazy and one owned operand, and `axpby_into` from a
//!   lazy source: `UnsupportedOnDevice`, asserted below (#1874).
//! - CUDA `svd_compact`/`qr_compact`/`eigh_full` of a lazy adjoint operand:
//!   `UnsupportedOnDevice`, asserted below.
//! - Fibonacci and SU(2)_k have complex twists (`Scalar = Complex64`), so
//!   they have no CUDA methods (compile-time absent); every device-admitted
//!   twist is ±1.
//! - CUDA compact storage does not exist (`to_cuda` densifies), so the
//!   compact column is Host-only.
//! - Checked Generic and complex-coefficient rules have no CUDA methods
//!   (compile-time absent).

use std::sync::Arc;

use num_complex::Complex64;
use tenet::sector::{
    FermionParityFusionRule, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep, Z2Irrep,
};
use tenet::typed::{
    Direction, Eigh, Error, GradedSpace, HermitianTol, OperationError, Qr, Runtime, SectorSpectrum,
    Svd, TensorMap,
};

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

/// The Hermitian entry `e[i,j] + conj(e[j,i])`.
fn hermitian_entry(seed: f64, coupled: usize, i: usize, j: usize) -> Complex64 {
    entry(seed, coupled, i, j) + entry(seed, coupled, j, i).conj()
}

/// `alpha * x + beta * y` entrywise.
fn combo(
    alpha: Complex64,
    x: &[(f64, Complex64)],
    beta: Complex64,
    y: &[(f64, Complex64)],
) -> Vec<(f64, Complex64)> {
    x.iter()
        .zip(y)
        .map(|((d, x), (_, y))| (*d, alpha * x + beta * y))
        .collect()
}

/// Host rows for one rule and payload: dense, lazy adjoint and compact
/// operands of every reduction and elementwise operation against the hand
/// oracle, plus the admission errors. The CUDA rows reuse the same fixtures
/// through `$cuda`.
macro_rules! storage_rows {
    ($runtime:expr, $leg:expr, $coupled:expr, $dim:expr, $D:ty, $conv:expr, $cuda:expr) => {{
        let runtime: &Runtime = $runtime;
        let leg = $leg;
        let coupled = $coupled;
        let dim = $dim;
        let conv = $conv;
        let build_with = |seed: f64, entry: fn(f64, usize, usize, usize) -> Complex64| {
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
        let build = |seed: f64| build_with(seed, entry);
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

        // Elementwise. `c` is on `a`'s space, `lazy_c` on its adjoint space;
        // for `t = e^H`, `<c^H, e^H> = conj(<c, e>)` and `norm(t) = norm(e)`.
        let (c, c_entries) = build(0.3);
        let lazy_c = c.adjoint().unwrap();
        let (alpha, beta) = (
            conv(Complex64::new(0.7, -0.3)),
            conv(Complex64::new(-1.1, 0.4)),
        );
        let (wa, wb) = (Complex64::from(alpha), Complex64::from(beta));
        let zero = Complex64::new(0.0, 0.0);
        let check = |what: &str, t: &TensorMap<_, $D>, e: &[(f64, Complex64)]| {
            let functional = Complex64::from(c.inner(t).unwrap());
            assert!(close_c(functional, inner_oracle(&c_entries, e)), "{what}");
            assert!(close(t.norm(2.0).unwrap(), norm_oracle(e, 2.0)), "{what} norm");
        };
        let check_lazy = |what: &str, t: &TensorMap<_, $D>, e: &[(f64, Complex64)]| {
            let functional = Complex64::from(lazy_c.inner(t).unwrap());
            assert!(close_c(functional, inner_oracle(&c_entries, e).conj()), "{what}");
            assert!(close(t.norm(2.0).unwrap(), norm_oracle(e, 2.0)), "{what} norm");
        };
        let ab = combo(wa, &a_entries, wb, &b_entries);
        let zeros = combo(zero, &a_entries, zero, &b_entries);
        check("scale", &a.scale(alpha).unwrap(), &combo(wa, &a_entries, zero, &b_entries));
        check("axpby", &a.axpby(alpha, &b, beta).unwrap(), &ab);
        check("zeros_like", &a.zeros_like().unwrap(), &zeros);
        let (mut destination, _) = build(-0.4);
        a.axpby_into(&mut destination, alpha, beta).unwrap();
        check("axpby_into", &destination, &ab);

        // Lazy adjoints: `alpha A^H + beta B^H = (conj(alpha) A + conj(beta) B)^H`.
        let lazy_ab = combo(wa.conj(), &a_entries, wb.conj(), &b_entries);
        let lazy_scaled = lazy_a.scale(alpha).unwrap();
        check_lazy("lazy scale", &lazy_scaled, &combo(wa.conj(), &a_entries, zero, &b_entries));
        let lazy_sum = lazy_a.axpby(alpha, &lazy_b, beta).unwrap();
        check_lazy("lazy + lazy", &lazy_sum, &lazy_ab);
        // Both stay lazy adjoints (A10): the payload is the parents'.
        for lazy in [&lazy_scaled, &lazy_sum] {
            assert!(matches!(lazy.dense_data(), Err(Error::Unsupported { .. })));
        }
        let owned_b = lazy_b.materialize().unwrap();
        check_lazy("lazy + owned", &lazy_a.axpby(alpha, &owned_b, beta).unwrap(), &lazy_ab);
        let zero_lazy = lazy_a.zeros_like().unwrap();
        check_lazy("lazy zeros_like", &zero_lazy, &zeros);
        assert!(matches!(zero_lazy.dense_data(), Err(Error::Unsupported { .. })));
        let mut destination = lazy_b.materialize().unwrap();
        lazy_a.axpby_into(&mut destination, alpha, beta).unwrap();
        check_lazy("lazy axpby_into", &destination, &lazy_ab);

        // Compact: scale and compact + compact stay compact.
        let (wide_alpha, wide_beta) = (Complex64::from(alpha), Complex64::from(beta));
        let diag_scaled: Vec<_> = diag_entries.iter().map(|(d, x)| (*d, wide_alpha * x)).collect();
        assert!(close(compact.scale(alpha).unwrap().norm(2.0).unwrap(), norm_oracle(&diag_scaled, 2.0)));
        let diag_sum = combo(wide_alpha, &diag_entries, wide_beta, &diag_entries);
        let compact_sum = compact.axpby(alpha, &compact, beta).unwrap();
        assert!(close(compact_sum.norm(2.0).unwrap(), norm_oracle(&diag_sum, 2.0)));
        assert!(close(compact.zeros_like().unwrap().norm(2.0).unwrap(), 0.0));

        // Admission: a foreign space, before any write.
        assert!(matches!(
            a.axpby(alpha, &wider, beta),
            Err(Error::Operation(op)) if matches!(*op, OperationError::SpaceMismatch { .. })
        ));
        let mut foreign = wider.clone();
        assert!(matches!(
            a.axpby_into(&mut foreign, alpha, beta),
            Err(Error::Operation(op)) if matches!(*op, OperationError::SpaceMismatch { .. })
        ));

        // Compact factors: `u s vh` and `q r` rebuild the operand, and `s`
        // and `r` carry its dim-weighted Frobenius norm (the outer factors are
        // isometries).
        let norm_a = norm_oracle(&a_entries, 2.0);
        let svd_rebuild = |Svd { u, s, vh }: Svd<TensorMap<_, $D>>| {
            assert!(close(s.norm(2.0).unwrap(), u.compose(&s).unwrap().norm(2.0).unwrap()));
            u.compose(&s).unwrap().compose(&vh).unwrap()
        };
        let qr_rebuild = |Qr { q, r }: Qr<TensorMap<_, $D>>| {
            assert!(close(r.norm(2.0).unwrap(), q.compose(&r).unwrap().norm(2.0).unwrap()));
            q.compose(&r).unwrap()
        };
        check("svd_compact", &svd_rebuild(a.svd_compact(&[0], &[1]).unwrap()), &a_entries);
        check("qr_compact", &qr_rebuild(a.qr_compact(&[0], &[1]).unwrap()), &a_entries);
        assert!(close(a.svd_compact(&[0], &[1]).unwrap().s.norm(2.0).unwrap(), norm_a));
        check_lazy(
            "lazy svd_compact",
            &svd_rebuild(lazy_a.svd_compact(&[0], &[1]).unwrap()),
            &a_entries,
        );
        check_lazy(
            "lazy qr_compact",
            &qr_rebuild(lazy_a.qr_compact(&[0], &[1]).unwrap()),
            &a_entries,
        );
        // A moved split factors `permute(a, [1], [0])`, of the same norm.
        let swapped = svd_rebuild(a.svd_compact(&[1], &[0]).unwrap());
        assert!(close(swapped.norm(2.0).unwrap(), norm_a));
        let swapped = qr_rebuild(a.qr_compact(&[1], &[0]).unwrap());
        assert!(close(swapped.norm(2.0).unwrap(), norm_a));
        let norm_diag = norm_oracle(&diag_entries, 2.0);
        let compact_svd = compact.svd_compact(&[0], &[1]).unwrap();
        assert!(close(compact_svd.s.norm(2.0).unwrap(), norm_diag));
        assert!(close(svd_rebuild(compact_svd).norm(2.0).unwrap(), norm_diag));
        let compact_qr = compact.qr_compact(&[0], &[1]).unwrap();
        assert!(close(qr_rebuild(compact_qr).norm(2.0).unwrap(), norm_diag));
        for malformed in [a.svd_compact(&[0], &[0]).map(|_| ()), a.qr_compact(&[0], &[2]).map(|_| ())] {
            assert!(matches!(
                malformed,
                Err(Error::Operation(op)) if matches!(*op, OperationError::InvalidPermutation { .. })
            ));
        }

        // Hermitian eigendecomposition: `v d v^H` rebuilds `h`, `d` carries
        // its norm (`v` is unitary), and `h^H = h` so the lazy adjoint
        // rebuilds `h` too.
        let tol = HermitianTol::DEFAULT;
        let (h, h_entries) = build_with(0.9, hermitian_entry);
        let lazy_h = h.adjoint().unwrap();
        let norm_h = norm_oracle(&h_entries, 2.0);
        let eigh_rebuild = |Eigh { d, v }: Eigh<TensorMap<_, $D>>| {
            assert!(close(d.norm(2.0).unwrap(), norm_h));
            v.compose(&d).unwrap().compose(&v.adjoint().unwrap()).unwrap()
        };
        check("eigh_full", &eigh_rebuild(h.eigh_full(&[0], &[1], tol).unwrap()), &h_entries);
        check("lazy eigh_full", &eigh_rebuild(lazy_h.eigh_full(&[0], &[1], tol).unwrap()), &h_entries);
        let swapped = eigh_rebuild(h.eigh_full(&[1], &[0], tol).unwrap());
        assert!(close(swapped.norm(2.0).unwrap(), norm_h));
        // A compact diagonal is Hermitian when its entries are real.
        let mut real_diag = Vec::new();
        let spectra: Vec<_> = leg
            .sectors()
            .unwrap()
            .into_iter()
            .map(|sector| {
                let k = leg.degeneracy(&sector).unwrap();
                let c = coupled(&sector);
                let values: Vec<$D> =
                    (0..k).map(|i| conv(Complex64::from(entry(0.6, c, i, i).re))).collect();
                real_diag.extend(values.iter().map(|&x| (dim(c), Complex64::from(x))));
                SectorSpectrum { sector, values }
            })
            .collect();
        let real_compact = TensorMap::<_, $D>::diagonal(runtime, &leg, spectra).unwrap();
        let Eigh { d, v } = real_compact.eigh_full(&[0], &[1], tol).unwrap();
        let norm_real = norm_oracle(&real_diag, 2.0);
        assert!(close(d.norm(2.0).unwrap(), norm_real));
        let rebuilt = v.compose(&d).unwrap().compose(&v.adjoint().unwrap()).unwrap();
        assert!(close(rebuilt.norm(2.0).unwrap(), norm_real));
        assert!(close_c(
            Complex64::from(real_compact.inner(&rebuilt).unwrap()),
            inner_oracle(&real_diag, &real_diag),
        ));
        // `a` is not Hermitian; malformed axes are the permute's error.
        assert!(matches!(a.eigh_full(&[0], &[1], tol), Err(Error::Operation(_))));
        assert!(matches!(
            h.eigh_full(&[0], &[0], tol),
            Err(Error::Operation(op)) if matches!(*op, OperationError::InvalidPermutation { .. })
        ));

        $cuda(&a, &b, &lazy_a, &h, alpha, beta);
    }};
}

fn no_cuda<T, D>(_: &T, _: &T, _: &T, _: &T, _: D, _: D) {}

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
fn host_rows_match_the_hand_oracle() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let real = |x: Complex64| x.re;
    let complex = |x: Complex64| x;
    let one = |_: usize| 1.0;
    let su2_dim = |twice: usize| (twice + 1) as f64;
    storage_rows!(&runtime, u1_leg(), u1_coupled, one, f64, real, no_cuda);
    storage_rows!(
        &runtime,
        u1_leg(),
        u1_coupled,
        one,
        Complex64,
        complex,
        no_cuda
    );
    storage_rows!(
        &runtime,
        su2_leg(),
        su2_coupled,
        su2_dim,
        f64,
        real,
        no_cuda
    );
    storage_rows!(
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
fn cuda_rows_match_host() {
    use tenet::sector::{CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec};
    use tenet::typed::{CudaFactorizationPayload, CudaStorage};

    fn device_rows<R, D>(
        a: &TensorMap<R, D>,
        b: &TensorMap<R, D>,
        lazy_a: &TensorMap<R, D>,
        h: &TensorMap<R, D>,
        alpha: D,
        beta: D,
    ) where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: CudaFactorizationPayload + Into<Complex64> + std::fmt::Debug,
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

        // Elementwise: the device result equals the Host result.
        let same = |what: &str, device: TensorMap<R, D, CudaStorage<D>>, host: TensorMap<R, D>| {
            let device = device.to_host().unwrap().materialize().unwrap();
            let host = host.materialize().unwrap();
            assert_eq!(device.codomain(), host.codomain(), "{what}");
            assert_eq!(device.domain(), host.domain(), "{what}");
            let (device, host) = (device.dense_data().unwrap(), host.dense_data().unwrap());
            assert_eq!(device.len(), host.len(), "{what}");
            for (&x, &y) in device.iter().zip(host) {
                assert!(close_c(x.into(), y.into()), "{what}");
            }
        };
        let lazy_dev = a_dev.adjoint().unwrap();
        let lazy_b = b.adjoint().unwrap();
        let lazy_b_dev = b_dev.adjoint().unwrap();
        same(
            "scale",
            a_dev.scale(alpha).unwrap(),
            a.scale(alpha).unwrap(),
        );
        same(
            "axpby",
            a_dev.axpby(alpha, &b_dev, beta).unwrap(),
            a.axpby(alpha, b, beta).unwrap(),
        );
        same(
            "zeros_like",
            a_dev.zeros_like().unwrap(),
            a.zeros_like().unwrap(),
        );
        let mut destination = b_dev.materialize().unwrap();
        a_dev.axpby_into(&mut destination, alpha, beta).unwrap();
        let mut expected = b.materialize().unwrap();
        a.axpby_into(&mut expected, alpha, beta).unwrap();
        same("axpby_into", destination, expected);
        same(
            "lazy scale",
            lazy_dev.scale(alpha).unwrap(),
            lazy_a.scale(alpha).unwrap(),
        );
        same(
            "lazy + lazy",
            lazy_dev.axpby(alpha, &lazy_b_dev, beta).unwrap(),
            lazy_a.axpby(alpha, &lazy_b, beta).unwrap(),
        );
        same(
            "lazy zeros_like",
            lazy_dev.zeros_like().unwrap(),
            lazy_a.zeros_like().unwrap(),
        );
        let owned_b = lazy_b.materialize().unwrap().to_cuda().unwrap();
        assert_eq!(
            lazy_dev.axpby(alpha, &owned_b, beta).unwrap_err(),
            Error::UnsupportedOnDevice(
                "axpby does not support mixed owned/lazy CUDA operands".into()
            )
        );
        let lazy_source = "axpby_into does not support lazy adjoint CUDA operands";
        let mut destination = lazy_b.materialize().unwrap().to_cuda().unwrap();
        assert_eq!(
            lazy_dev
                .axpby_into(&mut destination, alpha, beta)
                .unwrap_err(),
            Error::UnsupportedOnDevice(lazy_source.into())
        );
        // The destination checks run before the storage capability (#1756 L2).
        let shared = destination.clone();
        assert_eq!(
            lazy_dev
                .axpby_into(&mut destination, alpha, beta)
                .unwrap_err(),
            Error::DestinationShared
        );
        drop(shared);

        // Compact factors: the device rebuilds, `s` and `r` equal the Host's
        // (both storages fix the Host gauge), and a lazy device operand is
        // the storage's capability error.
        for (rows, cols) in [([0], [1]), ([1], [0])] {
            let (device, host) = (
                a_dev.svd_compact(&rows, &cols).unwrap(),
                a.svd_compact(&rows, &cols).unwrap(),
            );
            same("svd_compact s", device.s.clone(), host.s.clone());
            same(
                "svd_compact",
                device
                    .u
                    .compose(&device.s)
                    .unwrap()
                    .compose(&device.vh)
                    .unwrap(),
                host.u.compose(&host.s).unwrap().compose(&host.vh).unwrap(),
            );
            let (device, host) = (
                a_dev.qr_compact(&rows, &cols).unwrap(),
                a.qr_compact(&rows, &cols).unwrap(),
            );
            same("qr_compact r", device.r.clone(), host.r.clone());
            same(
                "qr_compact",
                device.q.compose(&device.r).unwrap(),
                host.q.compose(&host.r).unwrap(),
            );
        }
        assert_eq!(
            lazy_dev.svd_compact(&[0], &[1]).unwrap_err(),
            Error::UnsupportedOnDevice(
                "svd_compact does not support lazy adjoint CUDA operands".into()
            )
        );
        assert_eq!(
            lazy_dev.qr_compact(&[0], &[1]).unwrap_err(),
            Error::UnsupportedOnDevice(
                "qr_compact does not support lazy adjoint CUDA operands".into()
            )
        );

        // Hermitian eigendecomposition: the device `d` (dense) and `v d v^H`
        // equal the Host's.
        let tol = HermitianTol::DEFAULT;
        let h_dev = h.to_cuda().unwrap();
        for (rows, cols) in [([0], [1]), ([1], [0])] {
            let (device, host) = (
                h_dev.eigh_full(&rows, &cols, tol).unwrap(),
                h.eigh_full(&rows, &cols, tol).unwrap(),
            );
            same("eigh_full d", device.d.clone(), host.d.clone());
            same(
                "eigh_full",
                device
                    .v
                    .compose(&device.d)
                    .unwrap()
                    .compose(&device.v.adjoint().unwrap())
                    .unwrap(),
                host.v
                    .compose(&host.d)
                    .unwrap()
                    .compose(&host.v.adjoint().unwrap())
                    .unwrap(),
            );
        }
        assert_eq!(
            h_dev
                .adjoint()
                .unwrap()
                .eigh_full(&[0], &[1], tol)
                .unwrap_err(),
            Error::UnsupportedOnDevice(
                "eigh_full does not support lazy adjoint CUDA operands".into()
            )
        );
    }

    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let real = |x: Complex64| x.re;
    let complex = |x: Complex64| x;
    let one = |_: usize| 1.0;
    let su2_dim = |twice: usize| (twice + 1) as f64;
    storage_rows!(&runtime, u1_leg(), u1_coupled, one, f64, real, device_rows);
    storage_rows!(
        &runtime,
        u1_leg(),
        u1_coupled,
        one,
        Complex64,
        complex,
        device_rows
    );
    storage_rows!(
        &runtime,
        su2_leg(),
        su2_coupled,
        su2_dim,
        f64,
        real,
        device_rows
    );
    storage_rows!(
        &runtime,
        su2_leg(),
        su2_coupled,
        su2_dim,
        Complex64,
        complex,
        device_rows
    );
}

fn fz2_leg() -> GradedSpace<FermionParityFusionRule> {
    GradedSpace::try_new(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 3)],
    )
    .unwrap()
}

const TWIST_LEGS: [&[usize]; 5] = [&[0], &[1], &[2], &[0, 2], &[0, 1, 1]];

/// `twist` Host rows on fZ2 `[V, V*] <- [V]`: dense, lazy adjoint and compact
/// operands in both directions against `Σ θ |t|²`, plus the admission rows.
fn twist_rows<D>(runtime: &Runtime, conv: fn(Complex64) -> D)
where
    D: tenet::typed::TensorScalar + Into<Complex64>,
{
    let leg = fz2_leg();
    let dual = leg.try_dual().unwrap();
    // Per written entry: |x|² and the oddness of each leg's uncoupled sector.
    let mut entries: Vec<(f64, [bool; 3])> = Vec::new();
    let a = TensorMap::<_, D>::from_subblock_fn(runtime, [&leg, &dual], [&leg], |trees, ix| {
        let odd = |s: &Z2Irrep| *s == Z2Irrep::ODD;
        let (co, dom) = (trees.codomain_uncoupled(), trees.domain_uncoupled());
        let x = conv(entry(0.7, ix[0] + 2 * ix[1], ix[1], ix[2]));
        let weight = Into::<Complex64>::into(x).norm_sqr();
        entries.push((weight, [odd(&co[0]), odd(&co[1]), odd(&dom[0])]));
        x
    })
    .unwrap();
    let oracle = |legs: &[usize], parity: &dyn Fn(usize) -> usize| -> f64 {
        entries
            .iter()
            .map(|(weight, odd)| {
                let flips = legs.iter().filter(|&&l| odd[parity(l)]).count();
                if flips % 2 == 0 {
                    *weight
                } else {
                    -weight
                }
            })
            .sum()
    };
    let lazy = a.adjoint().unwrap();
    // `a^H` is `[V] <- [V, V*]`: logical leg 0 is parent leg 2, 1 and 2 are 0, 1.
    let to_parent = |l: usize| [2, 0, 1][l];
    for legs in TWIST_LEGS {
        let expected = oracle(legs, &|l| l);
        let expected_lazy = oracle(legs, &to_parent);
        assert!(
            expected.abs() < oracle(&[], &|l| l) - 1e-9,
            "{legs:?}: a block is negated"
        );
        for direction in [Direction::Forward, Direction::Inverse] {
            let twisted = a.twist(legs, direction).unwrap();
            assert!(close_c(a.inner(&twisted).unwrap().into(), expected.into()));
            let twisted = lazy.twist(legs, direction).unwrap();
            assert!(close_c(
                lazy.inner(&twisted).unwrap().into(),
                expected_lazy.into()
            ));
        }
    }
    // Compact: θ(c)^|legs| per sector, so an even leg count is the identity.
    let mut weights = Vec::new();
    let spectra: Vec<_> = leg
        .sectors()
        .unwrap()
        .into_iter()
        .map(|sector| {
            let sign = if sector == Z2Irrep::ODD { -1.0 } else { 1.0 };
            let values: Vec<D> = (0..leg.degeneracy(&sector).unwrap())
                .map(|i| conv(entry(0.3, i, i, i)))
                .collect();
            weights.extend(
                values
                    .iter()
                    .map(|&x| (sign, Into::<Complex64>::into(x).norm_sqr())),
            );
            tenet::typed::SectorSpectrum { sector, values }
        })
        .collect();
    let compact = TensorMap::<_, D>::diagonal(runtime, &leg, spectra).unwrap();
    for (legs, odd) in [(&[0usize][..], true), (&[1], true), (&[0, 1], false)] {
        let expected: f64 = weights
            .iter()
            .map(|&(sign, w)| if odd { sign * w } else { w })
            .sum();
        let twisted = compact.twist(legs, Direction::Forward).unwrap();
        assert!(twisted.dense_data().is_err(), "compact stays compact");
        assert!(close_c(
            compact.inner(&twisted).unwrap().into(),
            expected.into()
        ));
    }
    // Admission: the range check precedes the empty-list short circuit.
    assert!(matches!(
        a.twist(&[3], Direction::Forward),
        Err(Error::InvalidArgument(message)) if message == "twist leg 3 out of range for rank 3"
    ));
    assert!(matches!(
        lazy.twist(&[5], Direction::Inverse),
        Err(Error::InvalidArgument(message)) if message == "inverse twist leg 5 out of range for rank 3"
    ));
    let same = a.twist(&[], Direction::Forward).unwrap();
    assert!(close_c(
        a.inner(&same).unwrap().into(),
        oracle(&[], &|l| l).into()
    ));
}

#[test]
fn twist_host_rows_match_the_hand_oracle() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    twist_rows::<f64>(&runtime, |x| x.re);
    twist_rows::<Complex64>(&runtime, |x| x);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn twist_cuda_rows_match_host() {
    fn rows<D>(runtime: &Runtime, conv: fn(Complex64) -> D)
    where
        D: tenet::typed::CudaPayload + Into<Complex64> + std::fmt::Debug,
    {
        let leg = fz2_leg();
        let dual = leg.try_dual().unwrap();
        let a = TensorMap::<_, D>::from_subblock_fn(runtime, [&leg, &dual], [&leg], |_, ix| {
            conv(entry(0.7, ix[0] + 2 * ix[1], ix[1], ix[2]))
        })
        .unwrap();
        let a_dev = a.to_cuda().unwrap();
        for legs in TWIST_LEGS {
            for direction in [Direction::Forward, Direction::Inverse] {
                for (host, device) in [
                    (a.clone(), a_dev.clone()),
                    (a.adjoint().unwrap(), a_dev.adjoint().unwrap()),
                ] {
                    let device = device.twist(legs, direction).unwrap().to_host().unwrap();
                    let host = host.twist(legs, direction).unwrap();
                    let (device, host) =
                        (device.materialize().unwrap(), host.materialize().unwrap());
                    assert_eq!(device.codomain(), host.codomain(), "{legs:?}");
                    assert_eq!(device.domain(), host.domain(), "{legs:?}");
                    assert_eq!(
                        device.dense_data().unwrap(),
                        host.dense_data().unwrap(),
                        "{legs:?} {direction:?}: every device twist is ±1, so exact"
                    );
                }
            }
        }
        assert!(matches!(
            a_dev.twist(&[3], Direction::Forward),
            Err(Error::InvalidArgument(_))
        ));
    }
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    rows::<f64>(&runtime, |x| x.re);
    rows::<Complex64>(&runtime, |x| x);
}
