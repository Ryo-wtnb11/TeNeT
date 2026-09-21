//! Device twins of the `f64`/`Complex64` device *factorization* suites, run at
//! every admitted device payload (leaf C4, issue #1341).
//!
//! Each gate is one generic body instantiated for the dtypes the marker under
//! test admits, so the double-precision instantiation is the control: a helper
//! that silently did nothing would fail against `f64` first.
//!
//! **Oracle.** The host factorization of the *same tensor at the same payload
//! dtype*, read through gauge-independent identities — reconstruction,
//! isometry, the spectrum, the positive-diagonal gauge — never through a
//! payload comparison of a factor whose gauge the device does not promise.
//! Device `svd_compact` and `eigh_full` keep the raw cuSOLVER gauge, so `u`,
//! `vh` and the eigenvectors are compared only through those identities;
//! `qr_compact` *is* gauge-fixed on device (positive diagonal), so its factors
//! are compared pointwise. The host single-precision factorization semantics
//! themselves are gated independently by
//! `tenet/tests/single_precision_factorizations.rs` (#1324).
//!
//! **Tolerance.** `K * sqrt(n) * eps(real(D)) * scale * kappa`, with `K` the
//! constant of the shared single-precision oracle module, `n` the number of
//! payload entries the kernel sums over, `scale` stated per identity, and
//! `kappa = sigma_max / sigma_min` **measured from the double-precision
//! spectrum of the same fixture** — the same conditioning factor the host
//! suite uses. `eps` is the payload's own, so the `f64` instantiation is a
//! real control and not a vacuously loose one. No absolute platform constant
//! appears anywhere in this file.
//!
//! Fixture entries are dyadic rationals, so every fixture is exactly
//! representable in `f32` and the double-precision twin that measures `kappa`
//! holds exactly the widening of it.
//!
//! Run with `cargo test -p tenet-rs --features cuda,cpu-faer --test \
//! typed_cuda_single_precision_factorizations -- --ignored --test-threads=1`
//! on a CUDA host. The cost gate reads the process-wide transfer counters,
//! hence `--test-threads=1`.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use num_complex::{Complex32, Complex64};

use tenet::core::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use tenet::dense::{cuda_transfer_stats, CudaScalar};
use tenet::typed::{
    CudaFactorizationPayload, CudaQrPayload, CudaStorage, Error, GradedSpace, Runtime, TensorMap,
    Truncation,
};

mod common;
mod single_precision_oracle;

use common::{DevicePayload, DeviceRule};
use single_precision_oracle::{fermion_su2_leg_with, K};

// ---------------------------------------------------------------------------
// Payload markers
// ---------------------------------------------------------------------------

/// A device payload admitted to the device factorization family.
trait FactorPayload: DevicePayload + CudaFactorizationPayload {}
impl<D> FactorPayload for D where D: DevicePayload + CudaFactorizationPayload {}

/// A device payload additionally admitted to device QR (real payloads).
trait QrPayload: FactorPayload + CudaQrPayload {}
impl<D> QrPayload for D where D: FactorPayload + CudaQrPayload {}

// ---------------------------------------------------------------------------
// Tolerances
// ---------------------------------------------------------------------------

/// `K * sqrt(terms) * eps(real(D)) * max(scale, 1) * kappa`.
fn tolerance<D: DevicePayload>(terms: usize, scale: f64, kappa: f64) -> f64 {
    K * (terms as f64).sqrt() * D::EPS * scale.max(1.0) * kappa
}

fn assert_close<D: DevicePayload>(actual: &[D], expected: &[D], bound: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what} [{}]: length", D::NAME);
    for (index, (&left, &right)) in actual.iter().zip(expected).enumerate() {
        assert!(
            left.distance(right) <= bound,
            "{what} [{}]: entry {index} is {left:?}, expected {right:?} \
             (distance {}, bound {bound:e})",
            D::NAME,
            left.distance(right)
        );
    }
}

/// `||actual - expected||` over the whole tensor, against an absolute bound.
fn assert_residual<R, D>(
    actual: &TensorMap<R, D>,
    expected: &TensorMap<R, D>,
    bound: f64,
    what: &str,
) where
    R: DeviceRule,
    D: FactorPayload,
{
    let residual = actual
        .add(expected, D::entry(1.0, 0.0), D::entry(-1.0, 0.0))
        .unwrap()
        .norm()
        .unwrap();
    assert!(
        residual <= bound,
        "{what} [{}]: residual {residual:e} exceeds bound {bound:e}",
        D::NAME
    );
}

/// Block geometry and fusion trees, which the device must reproduce exactly.
type Structure<S> = Vec<(
    tenet::typed::BlockFusionTrees<S>,
    Vec<usize>,
    Vec<usize>,
    usize,
)>;

fn structure<R, D>(tensor: &TensorMap<R, D>) -> Structure<R::Sector>
where
    R: DeviceRule,
    D: FactorPayload,
{
    (0..tensor.block_count())
        .map(|index| {
            let block = tensor.block(index).unwrap();
            (
                tensor.block_fusion_trees(index).unwrap(),
                block.shape().to_vec(),
                block.strides().to_vec(),
                block.offset(),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn runtime() -> Runtime {
    Runtime::builder().cuda(0).dense_threads(1).build().unwrap()
}

/// Well-separated, distinct, positive: the default fixture diagonal.
const POSITIVE: [f64; 4] = [8.0, 4.0, 2.0, 1.0];

/// Eigenvalues of both signs whose *magnitudes* are not in the order the
/// values are, so a `lambda`-descending order and a `|lambda|`-descending one
/// disagree. cuSOLVER returns ascending `lambda`; only the second is TeNeT's
/// contract.
const INDEFINITE: [f64; 4] = [3.0, -1.0, 2.0, -4.0];

/// A well-separated diagonal with a small constant off-diagonal: distinct
/// singular values, Hermitian for a square block, and every value dyadic.
fn fixture_parts(diagonal: &[f64; 4], row: usize, col: usize) -> (f64, f64) {
    match row.cmp(&col) {
        std::cmp::Ordering::Equal => (diagonal[row % 4], 0.0),
        std::cmp::Ordering::Less => (0.25, 0.125),
        std::cmp::Ordering::Greater => (0.25, -0.125),
    }
}

fn fixture_with<R, D>(
    runtime: &Runtime,
    codomain: &GradedSpace<R>,
    domain: &GradedSpace<R>,
    diagonal: &[f64; 4],
) -> TensorMap<R, D>
where
    R: DeviceRule,
    D: FactorPayload,
{
    TensorMap::<R, D>::from_block_fn(runtime, [codomain], [domain], |_, index| {
        let (re, im) = fixture_parts(diagonal, index[0], index[1]);
        D::entry(re, im)
    })
    .unwrap()
}

fn fixture<R, D>(
    runtime: &Runtime,
    codomain: &GradedSpace<R>,
    domain: &GradedSpace<R>,
) -> TensorMap<R, D>
where
    R: DeviceRule,
    D: FactorPayload,
{
    fixture_with(runtime, codomain, domain, &POSITIVE)
}

/// `sigma_max / sigma_min` of the same fixture at double precision.
///
/// Measured through `Complex64` for every payload: a real fixture embedded in
/// the complex field has the same singular values, and the imaginary part is
/// included exactly when the payload under test carries one.
fn measured_kappa<R, D>(
    runtime: &Runtime,
    codomain: &GradedSpace<R>,
    domain: &GradedSpace<R>,
    diagonal: &[f64; 4],
) -> f64
where
    R: DeviceRule,
    D: FactorPayload,
{
    let wide =
        TensorMap::<R, Complex64>::from_block_fn(runtime, [codomain], [domain], |_, index| {
            let (re, im) = fixture_parts(diagonal, index[0], index[1]);
            Complex64::new(
                re,
                if <D as CudaScalar>::IS_COMPLEX {
                    im
                } else {
                    0.0
                },
            )
        })
        .unwrap();
    let mut largest = 0.0_f64;
    let mut smallest = f64::INFINITY;
    for entry in &wide.svd_vals().unwrap() {
        for &value in &entry.values {
            largest = largest.max(value);
            smallest = smallest.min(value);
        }
    }
    assert!(
        smallest > 0.0 && largest.is_finite(),
        "the fixture must have full rank in every coupled block"
    );
    largest / smallest
}

fn u1_leg(degeneracies: [usize; 3]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), degeneracies[0]),
            (U1Irrep::new(0), degeneracies[1]),
            (U1Irrep::new(1), degeneracies[2]),
        ],
    )
    .unwrap()
}

/// SU(2): non-Abelian, `dim(c) != 1`, several coupled sectors.
fn su2_leg() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// Compact SVD
// ---------------------------------------------------------------------------

fn assert_device_svd_matches_host<R, D>(
    runtime: &Runtime,
    codomain: &GradedSpace<R>,
    domain: &GradedSpace<R>,
) where
    R: DeviceRule,
    D: FactorPayload,
{
    let source = fixture::<R, D>(runtime, codomain, domain);
    let kappa = measured_kappa::<R, D>(runtime, codomain, domain, &POSITIVE);
    let terms = source.data().len().max(1);
    let norm = source.norm().unwrap();
    assert!(norm > 0.0, "the SVD fixture [{}] is vacuous", D::NAME);
    let bound = tolerance::<D>(terms, norm, kappa);

    let source_payload = source.data().to_vec();
    let (host_u, host_s, host_vh) = source.svd_compact().unwrap();

    let device = source.to_cuda().unwrap();
    let (device_u, device_s, device_vh) = device.svd_compact().unwrap();
    let provider = source.provider() as *const R;
    for factor in [&device_u, &device_s, &device_vh] {
        assert!(std::ptr::eq(factor.provider(), provider));
        assert_eq!(factor.placement(), tenet::core::Placement::Cuda(0));
    }

    let u = device_u.to_host().unwrap();
    let s = device_s.to_host().unwrap();
    let vh = device_vh.to_host().unwrap();
    for (actual, expected, what) in [
        (&u, &host_u, "svd u"),
        (&s, &host_s, "svd s"),
        (&vh, &host_vh, "svd vh"),
    ] {
        assert_eq!(
            structure(actual),
            structure(expected),
            "{what} [{}]: structure",
            D::NAME
        );
    }

    // The spectrum is gauge independent, so it is a pointwise oracle.
    assert_close(s.data(), host_s.data(), bound, "svd spectrum");

    // Descending and non-negative within every coupled sector: the contract
    // the truncation composition below relies on.
    for entry in &s.diagview().unwrap() {
        let mut previous = f64::INFINITY;
        for (index, value) in entry.values.iter().enumerate() {
            let (re, im) = value.parts();
            assert!(
                im.abs() <= bound,
                "svd spectrum [{}]: sector {:?} value {index} is not real ({im:e})",
                D::NAME,
                entry.sector
            );
            assert!(
                re >= -bound,
                "svd spectrum [{}]: sector {:?} value {index} is negative ({re:e})",
                D::NAME,
                entry.sector
            );
            assert!(
                re <= previous + bound,
                "svd spectrum [{}]: sector {:?} is not descending at {index}",
                D::NAME,
                entry.sector
            );
            previous = re;
        }
    }

    // Isometry and reconstruction: the two identities that hold in any gauge.
    let identity_bound = tolerance::<D>(terms, 1.0, kappa);
    assert_residual(
        &u.adjoint().unwrap().compose(&u).unwrap(),
        &TensorMap::<R, D>::id(runtime, u.domain().iter()).unwrap(),
        identity_bound,
        "svd u isometry",
    );
    assert_residual(
        &vh.compose(&vh.adjoint().unwrap()).unwrap(),
        &TensorMap::<R, D>::id(runtime, vh.codomain().iter()).unwrap(),
        identity_bound,
        "svd vh coisometry",
    );
    assert_residual(
        &u.compose(&s).unwrap().compose(&vh).unwrap(),
        &source,
        bound,
        "svd reconstruction",
    );

    // The source is an input, not scratch: bit for bit unchanged.
    assert_eq!(device.to_host().unwrap().data(), source_payload);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_svd_compact_matches_the_host_at_every_payload() {
    let runtime = runtime();
    let square = u1_leg([2, 3, 2]);
    let tall = u1_leg([3, 4, 3]);
    let su2 = su2_leg();
    let fermion = fermion_su2_leg_with([2, 2, 1]);

    for (codomain, domain) in [(&square, &square), (&tall, &square)] {
        assert_device_svd_matches_host::<_, f64>(&runtime, codomain, domain);
        assert_device_svd_matches_host::<_, Complex64>(&runtime, codomain, domain);
        assert_device_svd_matches_host::<_, f32>(&runtime, codomain, domain);
        assert_device_svd_matches_host::<_, Complex32>(&runtime, codomain, domain);
    }
    // Wide as well as tall, so both compact factor shapes are assembled.
    assert_device_svd_matches_host::<_, f32>(&runtime, &square, &tall);
    assert_device_svd_matches_host::<_, Complex32>(&runtime, &square, &tall);

    // SU(2): several quantum dimensions. fZ2 x U(1) x SU(2): fermionic signs
    // and non-Abelian recoupling in one provider.
    assert_device_svd_matches_host::<_, f32>(&runtime, &su2, &su2);
    assert_device_svd_matches_host::<_, Complex32>(&runtime, &su2, &su2);
    assert_device_svd_matches_host::<_, f32>(&runtime, &fermion, &fermion);
    assert_device_svd_matches_host::<_, Complex32>(&runtime, &fermion, &fermion);
}

// ---------------------------------------------------------------------------
// Compact QR (real payloads only)
// ---------------------------------------------------------------------------

fn assert_device_qr_matches_host<R, D>(
    runtime: &Runtime,
    codomain: &GradedSpace<R>,
    domain: &GradedSpace<R>,
) where
    R: DeviceRule,
    D: QrPayload,
{
    let source = fixture::<R, D>(runtime, codomain, domain);
    let kappa = measured_kappa::<R, D>(runtime, codomain, domain, &POSITIVE);
    let terms = source.data().len().max(1);
    let norm = source.norm().unwrap();
    let bound = tolerance::<D>(terms, norm, kappa);

    let (host_q, host_r) = source.qr_compact().unwrap();
    let device = source.to_cuda().unwrap();
    let (device_q, device_r) = device.qr_compact().unwrap();
    let q = device_q.to_host().unwrap();
    let r = device_r.to_host().unwrap();

    assert_eq!(structure(&q), structure(&host_q), "qr q [{}]", D::NAME);
    assert_eq!(structure(&r), structure(&host_r), "qr r [{}]", D::NAME);

    // QR *is* gauge-fixed on device (positive diagonal), so — unlike the SVD —
    // host and device factors are the same object and compare pointwise.
    assert_close(q.data(), host_q.data(), bound, "qr q");
    assert_close(r.data(), host_r.data(), bound, "qr r");

    assert_residual(
        &q.adjoint().unwrap().compose(&q).unwrap(),
        &TensorMap::<R, D>::id(runtime, q.domain().iter()).unwrap(),
        tolerance::<D>(terms, 1.0, kappa),
        "qr q isometry",
    );
    assert_residual(&q.compose(&r).unwrap(), &source, bound, "qr reconstruction");
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_qr_compact_matches_the_host_at_every_real_payload() {
    let runtime = runtime();
    let square = u1_leg([2, 3, 2]);
    let tall = u1_leg([3, 4, 3]);
    let su2 = su2_leg();
    let fermion = fermion_su2_leg_with([2, 2, 1]);

    for (codomain, domain) in [(&square, &square), (&tall, &square), (&square, &tall)] {
        assert_device_qr_matches_host::<_, f64>(&runtime, codomain, domain);
        assert_device_qr_matches_host::<_, f32>(&runtime, codomain, domain);
    }
    assert_device_qr_matches_host::<_, f32>(&runtime, &su2, &su2);
    assert_device_qr_matches_host::<_, f32>(&runtime, &fermion, &fermion);
}

/// The positive-diagonal gauge the backend applies on device, read through the
/// public per-sector diagonal of the `bond <- bond` triangular factor.
///
/// The positivity is exact: it is the sign the gauge step applies, not a
/// computed quantity. Reality of the diagonal is bounded, not exact, because
/// the sign is applied as a multiplication.
#[test]
#[ignore = "requires a real CUDA device"]
fn device_qr_returns_the_positive_diagonal_gauge_at_every_real_payload() {
    fn assert_gauge<D: QrPayload>(runtime: &Runtime, leg: &GradedSpace<U1FusionRule>) {
        let source = fixture::<U1FusionRule, D>(runtime, leg, leg);
        let terms = source.data().len().max(1);
        let (_, r) = source.to_cuda().unwrap().qr_compact().unwrap();
        let spectra = r.to_host().unwrap().diagview().unwrap();
        assert!(
            !spectra.is_empty(),
            "the triangular factor [{}] has no coupled blocks",
            D::NAME
        );
        for entry in &spectra {
            for value in &entry.values {
                let (re, im) = value.parts();
                let bound = tolerance::<D>(terms, re.abs(), 1.0);
                assert!(
                    re > 0.0 && im.abs() <= bound,
                    "qr R diagonal [{}]: sector {:?} entry {re}+{im}i is not positive real \
                     within {bound:e}",
                    D::NAME,
                    entry.sector
                );
            }
        }
    }

    let runtime = runtime();
    let leg = u1_leg([2, 3, 2]);
    assert_gauge::<f64>(&runtime, &leg);
    assert_gauge::<f32>(&runtime, &leg);
}

// ---------------------------------------------------------------------------
// Hermitian eigendecomposition
// ---------------------------------------------------------------------------

fn assert_device_eigh_matches_host<R, D>(
    runtime: &Runtime,
    leg: &GradedSpace<R>,
    diagonal: &[f64; 4],
) where
    R: DeviceRule,
    D: FactorPayload,
{
    let source = fixture_with::<R, D>(runtime, leg, leg, diagonal);
    let kappa = measured_kappa::<R, D>(runtime, leg, leg, diagonal);
    let terms = source.data().len().max(1);
    let norm = source.norm().unwrap();
    let bound = tolerance::<D>(terms, norm, kappa);

    let (host_d, _host_v) = source.eigh_full().unwrap();
    let device = source.to_cuda().unwrap();
    let (device_d, device_v) = device.eigh_full().unwrap();
    let d = device_d.to_host().unwrap();
    let v = device_v.to_host().unwrap();

    assert_eq!(structure(&d), structure(&host_d), "eigh d [{}]", D::NAME);

    // Eigenvalues are gauge independent up to the documented |lambda|
    // ordering, which both sides share, so they compare pointwise.
    assert_close(d.data(), host_d.data(), bound, "eigh spectrum");

    // |lambda| descending within every coupled sector.
    for entry in &d.diagview().unwrap() {
        let mut previous = f64::INFINITY;
        for (index, value) in entry.values.iter().enumerate() {
            let magnitude = value.magnitude();
            assert!(
                magnitude <= previous + bound,
                "eigh spectrum [{}]: sector {:?} is not |lambda|-descending at {index}",
                D::NAME,
                entry.sector
            );
            previous = magnitude;
        }
    }

    assert_residual(
        &v.adjoint().unwrap().compose(&v).unwrap(),
        &TensorMap::<R, D>::id(runtime, v.domain().iter()).unwrap(),
        tolerance::<D>(terms, 1.0, kappa),
        "eigh v isometry",
    );
    assert_residual(
        &v.compose(&d)
            .unwrap()
            .compose(&v.adjoint().unwrap())
            .unwrap(),
        &source,
        bound,
        "eigh reconstruction",
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_eigh_full_matches_the_host_at_every_payload() {
    let runtime = runtime();
    let u1 = u1_leg([2, 3, 2]);
    let su2 = su2_leg();
    let fermion = fermion_su2_leg_with([2, 2, 1]);

    // Both the positive-definite and the indefinite fixture: the second is
    // what separates the `|lambda|`-descending contract from cuSOLVER's own
    // ascending-`lambda` order.
    for diagonal in [&POSITIVE, &INDEFINITE] {
        assert_device_eigh_matches_host::<_, f64>(&runtime, &u1, diagonal);
        assert_device_eigh_matches_host::<_, Complex64>(&runtime, &u1, diagonal);
        assert_device_eigh_matches_host::<_, f32>(&runtime, &u1, diagonal);
        assert_device_eigh_matches_host::<_, Complex32>(&runtime, &u1, diagonal);
        assert_device_eigh_matches_host::<_, f32>(&runtime, &su2, diagonal);
        assert_device_eigh_matches_host::<_, Complex32>(&runtime, &su2, diagonal);
        assert_device_eigh_matches_host::<_, f32>(&runtime, &fermion, diagonal);
        assert_device_eigh_matches_host::<_, Complex32>(&runtime, &fermion, diagonal);
    }
}

/// The device Hermitian admission rule at the payload's own epsilon (#1326).
///
/// The perturbed fixture's relative anti-Hermitian residual sits between
/// `64 * eps(f64)` and `64 * eps(f32)`: an `f32` block is entitled to it and is
/// admitted, and the `f64` twin of the *same numbers* is not — which is the
/// whole point of typing the tolerance by the real lane. Before C1 the rule
/// compared every payload against `64 * eps(f64)` and would have rejected the
/// single-precision block.
///
/// The perturbation is a power of two added to a power of two, so it is exact
/// in `f32` and the `f64` twin holds its widening.
#[test]
#[ignore = "requires a real CUDA device"]
fn device_eigh_admits_a_nearly_hermitian_single_precision_block() {
    fn perturbed<D: FactorPayload>(
        runtime: &Runtime,
        leg: &GradedSpace<U1FusionRule>,
        skew: f64,
    ) -> TensorMap<U1FusionRule, D, CudaStorage<D>> {
        TensorMap::<U1FusionRule, D>::from_block_fn(runtime, [leg], [leg], |_, index| {
            let (re, im) = fixture_parts(&POSITIVE, index[0], index[1]);
            if index[0] > index[1] {
                D::entry(re + skew, im)
            } else {
                D::entry(re, im)
            }
        })
        .unwrap()
        .to_cuda()
        .unwrap()
    }

    let runtime = runtime();
    let leg = u1_leg([2, 3, 2]);
    // 2^-20 against a fixture of Frobenius norm >= 8: a relative residual near
    // 6e-8, inside 64*eps(f32) = 7.6e-6 and far outside 64*eps(f64) = 1.4e-14.
    let skew = (-20.0_f64).exp2();

    assert!(
        perturbed::<f32>(&runtime, &leg, skew).eigh_full().is_ok(),
        "an f32 block within 64*eps(f32) must be admitted"
    );
    assert!(
        perturbed::<Complex32>(&runtime, &leg, skew)
            .eigh_full()
            .is_ok(),
        "a Complex32 block within 64*eps(f32) must be admitted"
    );
    assert!(
        perturbed::<f64>(&runtime, &leg, skew).eigh_full().is_err(),
        "the same numbers at f64 exceed 64*eps(f64) and must be rejected"
    );

    // A genuinely non-Hermitian block is rejected at every payload, and the
    // rejection names the requirement rather than the storage.
    let gross = 0.5;
    for (name, error) in [
        (
            "f64",
            perturbed::<f64>(&runtime, &leg, gross).eigh_full().err(),
        ),
        (
            "c64",
            perturbed::<Complex64>(&runtime, &leg, gross)
                .eigh_full()
                .err(),
        ),
        (
            "f32",
            perturbed::<f32>(&runtime, &leg, gross).eigh_full().err(),
        ),
        (
            "c32",
            perturbed::<Complex32>(&runtime, &leg, gross)
                .eigh_full()
                .err(),
        ),
    ] {
        let message = format!("{error:?}");
        assert!(
            error.is_some() && message.contains("Hermitian"),
            "{name}: a non-Hermitian block must be rejected as such, got {message}"
        );
    }
}

// ---------------------------------------------------------------------------
// Rejection order
// ---------------------------------------------------------------------------

/// The capability and operand boundaries precede every device action, at every
/// payload, and in the documented order: the truncated entry points report the
/// missing capability even on a lazy-adjoint receiver whose storage would also
/// have been rejected.
#[test]
#[ignore = "requires a real CUDA device"]
fn device_factorization_rejections_do_not_depend_on_the_payload() {
    fn assert_rejections<D: FactorPayload>(runtime: &Runtime, leg: &GradedSpace<U1FusionRule>) {
        let device = fixture::<U1FusionRule, D>(runtime, leg, leg)
            .to_cuda()
            .unwrap();
        let lazy = device.adjoint().unwrap();

        let before = cuda_transfer_stats();
        for (operation, error) in [
            ("svd_compact", lazy.svd_compact().err()),
            ("eigh_full", lazy.eigh_full().err()),
        ] {
            assert!(
                matches!(&error, Some(Error::UnsupportedOnDevice(message))
                    if message.contains(operation) && message.contains("lazy adjoint")),
                "{operation} [{}]: {error:?}",
                D::NAME
            );
        }
        // The truncated variants are a *capability* boundary and win over the
        // operand check: a lazy-adjoint receiver still reports the capability.
        for (operation, error) in [
            ("svd_trunc", lazy.svd_trunc(&Truncation::rank(2)).err()),
            ("eigh_trunc", lazy.eigh_trunc(&Truncation::rank(2)).err()),
            ("svd_trunc", device.svd_trunc(&Truncation::rank(2)).err()),
            ("eigh_trunc", device.eigh_trunc(&Truncation::rank(2)).err()),
        ] {
            assert!(
                matches!(&error, Some(Error::UnsupportedOnDevice(message))
                    if message.contains(operation) && message.contains("find_truncated")),
                "{operation} [{}]: {error:?}",
                D::NAME
            );
        }
        // No lease, no plan, no allocation, no transfer.
        assert_eq!(
            cuda_transfer_stats(),
            before,
            "a rejected device factorization [{}] touched the device",
            D::NAME
        );
    }

    let runtime = runtime();
    let leg = u1_leg([2, 3, 2]);
    assert_rejections::<f64>(&runtime, &leg);
    assert_rejections::<Complex64>(&runtime, &leg);
    assert_rejections::<f32>(&runtime, &leg);
    assert_rejections::<Complex32>(&runtime, &leg);
}

// ---------------------------------------------------------------------------
// #1320: factor blocks at unaligned offsets
// ---------------------------------------------------------------------------

/// #1320 at 4- and 8-byte elements.
///
/// The aligned whole-factor route copies each block straight into its sector
/// region, and pinned Tenferro tells cuTENSOR that an offset destination view
/// is 256-byte aligned when it is not. Which offsets are affected depends on
/// the element size, so admitting `f32` (4 bytes) and `Complex32` (8 bytes)
/// changes the set: every degeneracy pair below puts the second sector's
/// block at an **odd** element offset (9, 25, 9), which is unaligned for all
/// four payload sizes at once. `(4, 2)` and `(2, 2)` are the controls.
#[test]
#[ignore = "requires a real CUDA device"]
fn device_factorizations_handle_blocks_at_unaligned_offsets_at_every_payload() {
    let runtime = runtime();
    let u1 = Arc::new(U1FusionRule);

    for (d0, d1) in [(3usize, 2usize), (5, 2), (3, 3), (4, 2), (2, 2)] {
        let leg = GradedSpace::try_new_with_arc(
            Arc::clone(&u1),
            [(U1Irrep::new(0), d0), (U1Irrep::new(1), d1)],
        )
        .unwrap();
        assert_device_svd_matches_host::<_, f32>(&runtime, &leg, &leg);
        assert_device_svd_matches_host::<_, Complex32>(&runtime, &leg, &leg);
        assert_device_svd_matches_host::<_, f64>(&runtime, &leg, &leg);
        assert_device_svd_matches_host::<_, Complex64>(&runtime, &leg, &leg);
        assert_device_qr_matches_host::<_, f32>(&runtime, &leg, &leg);
        assert_device_qr_matches_host::<_, f64>(&runtime, &leg, &leg);
    }
}

// ---------------------------------------------------------------------------
// #1297: the truncation composition at single precision
// ---------------------------------------------------------------------------

/// Device `svd_compact` -> `to_host` -> `diagview` -> `find_truncated` ->
/// `restrict_*` reproduces host `svd_trunc` **at the same dtype**.
///
/// The kept bond space is asserted exactly: the fixture's singular values are
/// well separated, so the rank budget is not at a tie and the documented
/// near-tie behaviour of #1324 does not apply. The discarded-weight
/// postcondition is compared within the spectrum's own tolerance.
fn assert_truncation_composition<R, D>(
    runtime: &Runtime,
    leg: &GradedSpace<R>,
    truncation: &Truncation,
) where
    R: DeviceRule,
    D: FactorPayload + tenet::typed::SpectrumMagnitude,
{
    let source = fixture::<R, D>(runtime, leg, leg);
    let kappa = measured_kappa::<R, D>(runtime, leg, leg, &POSITIVE);
    let terms = source.data().len().max(1);
    let bound = tolerance::<D>(terms, source.norm().unwrap(), kappa);

    let expected = source.svd_trunc(truncation).unwrap();

    let (device_u, device_s, device_vh) = source.to_cuda().unwrap().svd_compact().unwrap();
    let u = device_u.to_host().unwrap();
    let s = device_s.to_host().unwrap();
    let vh = device_vh.to_host().unwrap();

    let found = s.domain()[0]
        .find_truncated(&s.diagview().unwrap(), truncation)
        .unwrap();
    let selection = &found.selection;
    let u = u.restrict_leg(u.codomain_rank(), selection).unwrap();
    let s = s.restrict_diagonal(selection).unwrap();
    let vh = vh.restrict_leg(0, selection).unwrap();

    assert_eq!(
        *selection.subspace(),
        expected.s.domain()[0],
        "kept bond space [{}]",
        D::NAME
    );
    for (actual, expected, what) in [
        (&u, &expected.u, "u"),
        (&s, &expected.s, "s"),
        (&vh, &expected.vh, "vh"),
    ] {
        assert_eq!(
            structure(actual),
            structure(expected),
            "truncated {what} [{}]",
            D::NAME
        );
    }

    let mut kept = s.diagview().unwrap();
    kept.sort_by(|left, right| left.sector.cmp(&right.sector));
    assert_eq!(kept.len(), expected.singular_values.len());
    for (actual, oracle) in kept.iter().zip(&expected.singular_values) {
        assert_eq!(actual.sector, oracle.sector);
        assert_eq!(actual.values.len(), oracle.values.len());
        for (&value, &reference) in actual.values.iter().zip(&oracle.values) {
            let (re, im) = value.parts();
            assert!(
                (re - reference).abs() <= bound && im.abs() <= bound,
                "kept spectrum [{}]: {re}+{im}i against {reference}",
                D::NAME
            );
        }
    }
    assert!(
        (found.error - expected.error).abs() <= bound,
        "discarded weight [{}]: {} against {}",
        D::NAME,
        found.error,
        expected.error
    );

    // Reconstruction of the truncated triple, which is the postcondition a
    // caller composing this recipe actually relies on.
    assert_residual(
        &u.compose(&s).unwrap().compose(&vh).unwrap(),
        &expected
            .u
            .compose(&expected.s)
            .unwrap()
            .compose(&expected.vh)
            .unwrap(),
        bound,
        "truncated reconstruction",
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn the_device_truncation_composition_matches_host_svd_trunc_at_every_payload() {
    let runtime = runtime();
    let leg = u1_leg([2, 3, 2]);
    let su2 = su2_leg();

    // Every budget lands in a gap of the fixture's spectrum, so no `rank`
    // boundary falls on a tie and the kept space is exactly the host's.
    // Sector diagonals are `(8, 4)`, `(8, 4, 2)`, `(8, 4)` for U(1) and
    // `(8, 4)` twice for SU(2), whose quantum dimensions are `1` and `2`:
    // `3` keeps every `8` and `6` keeps every `8` and `4`. The near-tie
    // behaviour #1324 documents is therefore out of scope here by
    // construction, not by tolerance.
    for truncation in [Truncation::Full, Truncation::rank(3), Truncation::rank(6)] {
        assert_truncation_composition::<_, f64>(&runtime, &leg, &truncation);
        assert_truncation_composition::<_, Complex64>(&runtime, &leg, &truncation);
        assert_truncation_composition::<_, f32>(&runtime, &leg, &truncation);
        assert_truncation_composition::<_, Complex32>(&runtime, &leg, &truncation);
        assert_truncation_composition::<_, f32>(&runtime, &su2, &truncation);
        assert_truncation_composition::<_, Complex32>(&runtime, &su2, &truncation);
    }
}

// ---------------------------------------------------------------------------
// Cost contract
// ---------------------------------------------------------------------------

/// Same device work, half the bytes — for the factorizations.
///
/// Relative and same-process: the single-precision run of a fixture is
/// compared against the double-precision run of the *same* fixture in the same
/// test binary, so no absolute platform constant appears. Call counts must be
/// equal (the schedule does not depend on the payload dtype) and transferred
/// bytes exactly halved (the element size does).
#[test]
#[ignore = "requires a real CUDA device"]
fn device_factorizations_cost_the_same_calls_and_half_the_bytes() {
    type Cost = (u64, u64, u64, u64, u64, u64, u64);

    fn measure<D: FactorPayload>(runtime: &Runtime, leg: &GradedSpace<U1FusionRule>) -> Cost {
        let device = fixture::<U1FusionRule, D>(runtime, leg, leg)
            .to_cuda()
            .unwrap();
        // Warm the lane, the scalar operands and the kernels of this dtype.
        drop(device.svd_compact().unwrap());
        drop(device.eigh_full().unwrap());

        let before = cuda_transfer_stats();
        drop(device.svd_compact().unwrap());
        drop(device.eigh_full().unwrap());
        let after = cuda_transfer_stats();
        (
            after.h2d_calls - before.h2d_calls,
            after.h2d_bytes - before.h2d_bytes,
            after.d2h_calls - before.d2h_calls,
            after.d2h_bytes - before.d2h_bytes,
            after.device_allocs - before.device_allocs,
            after.copy_calls - before.copy_calls,
            after.gemm_calls - before.gemm_calls,
        )
    }

    let runtime = runtime();
    let leg = u1_leg([2, 3, 2]);

    for (narrow, wide, name) in [
        (
            measure::<f32>(&runtime, &leg),
            measure::<f64>(&runtime, &leg),
            "f32/f64",
        ),
        (
            measure::<Complex32>(&runtime, &leg),
            measure::<Complex64>(&runtime, &leg),
            "c32/c64",
        ),
    ] {
        eprintln!("factorization cost {name}: narrow {narrow:?} wide {wide:?}");
        assert_eq!(
            (narrow.0, narrow.2, narrow.4, narrow.5, narrow.6),
            (wide.0, wide.2, wide.4, wide.5, wide.6),
            "{name}: (h2d_calls, d2h_calls, device_allocs, copy_calls, gemm_calls) \
             must not depend on the payload dtype"
        );
        assert!(wide.1 > 0 && wide.3 > 0, "{name}: vacuous byte counts");
        assert_eq!(narrow.1 * 2, wide.1, "{name}: h2d bytes must be halved");
        assert_eq!(narrow.3 * 2, wide.3, "{name}: d2h bytes must be halved");
    }
}
