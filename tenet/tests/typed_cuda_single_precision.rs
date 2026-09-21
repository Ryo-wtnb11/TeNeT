//! Device twins of the `f64`/`Complex64` base-family device suites, run at
//! every admitted device payload (leaf C2, issue #1336).
//!
//! Each gate here is one generic body instantiated for all four dtypes, so the
//! double-precision instantiation is the control: a helper that silently did
//! nothing would fail against `f64` first. The oracle is always the **host**
//! result *at the same payload dtype* — never the `f64` host result — so what
//! these prove is that the device agrees with TeNeT's own host semantics at
//! the payload's own precision. The host single-precision semantics themselves
//! are gated independently by `tenet/tests/single_precision_base.rs` (#1315).
//!
//! Tolerances are stated per family, not shared:
//!
//! * transfer is **bit exact** at every dtype — a byte-for-byte round trip;
//! * elementwise arithmetic and the structural transforms are compared at
//!   `8 * sqrt(n) * eps(real(D)) * scale`, `n` the number of terms combined;
//! * the reductions get their own, wider bound, because the device sums a
//!   coupled sector *in the payload dtype* inside the GEMM while the host
//!   accumulates in `WideScalar::Wide`. That contract, and the overflow it
//!   implies, is documented on `weighted_inner_cuda` and pinned by
//!   `device_single_precision_norm_can_overflow_where_the_host_stays_finite`.
//!
//! Fixture entries are dyadic rationals, so every fixture is exactly
//! representable in `f32` and no fixture value is itself a rounding of the
//! double-precision one.
//!
//! Run with `cargo test -p tenet-rs --features cuda,cpu-faer --test \
//! typed_cuda_single_precision -- --ignored --test-threads=1` on a CUDA host.
//! The counter gate reads the process-wide transfer counters, hence
//! `--test-threads=1`.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use num_complex::{Complex32, Complex64};

use tenet::core::{
    product_sector, FermionParityFusionRule, ProductFusionRule, ProductFusionRuleExt,
    SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep, Z2Irrep,
};
use tenet::dense::cuda_transfer_stats;
use tenet::typed::{BlockFusionTrees, Error, GradedSpace, Runtime, TensorMap};

mod common;

use common::{DevicePayload, DeviceRule};

// ---------------------------------------------------------------------------
// Tolerances
// ---------------------------------------------------------------------------

/// Elementwise bound for a result that combined `terms` values of magnitude
/// `scale`: `8 * sqrt(terms) * eps(real(D)) * scale`.
fn elementwise_tolerance<D: DevicePayload>(terms: usize, scale: f64) -> f64 {
    8.0 * (terms as f64).sqrt() * D::EPS * scale.max(1.0)
}

/// Bound for a reduction the device accumulated *in the payload dtype*:
/// `2 * terms * eps(real(D)) * scale`, where the caller passes `scale` as an
/// upper bound on `sum |conj(a_i) * b_i|` — the sum of the **absolute**
/// products, which is what the documented device bound on
/// `TensorMap::<..., CudaStorage<D>>::norm` is relative to. It is not the
/// host's wide-accumulator tolerance, and it is not relative to the magnitude
/// of the result.
fn reduction_tolerance<D: DevicePayload>(terms: usize, scale: f64) -> f64 {
    2.0 * (terms as f64) * D::EPS * scale.max(1.0)
}

/// Compares against `tolerance * (1 + |expected|)`.
///
/// The relative factor belongs to the elementwise families, whose
/// `elementwise_tolerance` is a *relative* bound. The reductions use
/// [`assert_close_absolutely`] instead: their documented bound is already
/// absolute.
fn assert_close<D: DevicePayload>(actual: &[D], expected: &[D], tolerance: f64, what: &str) {
    assert_within(actual, expected, tolerance, what, true)
}

/// Compares against `tolerance` exactly, with no relative widening.
///
/// This is the bound `reduction_tolerance` documents, asserted as documented.
/// #1336 left the extra `(1 + |expected|)` factor in place here because
/// tightening a device tolerance cannot be validated without a device run;
/// leaf C4 (#1341) removed it on the A100 record of this suite.
fn assert_close_absolutely<D: DevicePayload>(
    actual: &[D],
    expected: &[D],
    tolerance: f64,
    what: &str,
) {
    assert_within(actual, expected, tolerance, what, false)
}

fn assert_within<D: DevicePayload>(
    actual: &[D],
    expected: &[D],
    tolerance: f64,
    what: &str,
    relative: bool,
) {
    assert_eq!(actual.len(), expected.len(), "{what} [{}]: length", D::NAME);
    for (index, (&left, &right)) in actual.iter().zip(expected).enumerate() {
        let bound = if relative {
            tolerance * (1.0 + right.magnitude())
        } else {
            tolerance
        };
        assert!(
            left.distance(right) <= bound,
            "{what} [{}]: element {index} is {left:?}, expected {right:?} \
             (distance {}, bound {bound})",
            D::NAME,
            left.distance(right)
        );
    }
}

fn assert_bit_exact<D: DevicePayload>(actual: &[D], expected: &[D], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what} [{}]: length", D::NAME);
    for (index, (left, right)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(left, right, "{what} [{}]: element {index}", D::NAME);
    }
}

/// Non-vacuity: the payload actually moved, so a device replay that wrote
/// nothing could not pass.
fn assert_moved<D: DevicePayload>(source: &[D], transformed: &[D], what: &str) {
    assert!(
        source.iter().zip(transformed).any(|(a, b)| a != b),
        "{what} [{}]: the operation left the payload identical",
        D::NAME
    );
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn runtime() -> Runtime {
    Runtime::builder().cuda(0).dense_threads(1).build().unwrap()
}

/// Dyadic, index-dependent, and complex-nontrivial where the dtype allows it.
fn fill<D: DevicePayload, S>(seed: f64) -> impl FnMut(&BlockFusionTrees<S>, &[usize]) -> D {
    let mut ordinal = 0.0_f64;
    move |_, indices| {
        ordinal += 1.0;
        let position: f64 = indices.iter().map(|&i| i as f64).sum();
        D::entry(
            seed + 0.25 * position + 0.125 * (ordinal % 7.0),
            0.5 - 0.25 * (ordinal % 5.0) + 0.125 * position,
        )
    }
}

fn u1_leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 1),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap()
}

fn su2_leg() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(2), 1),
        ],
    )
    .unwrap()
}

type FermionSu2 = ProductFusionRule<FermionParityFusionRule, SU2FusionRule>;

/// `fZ2 ⊠ SU(2)`: fermionic signs *and* non-Abelian recoupling in one provider.
fn fz2_su2_leg() -> GradedSpace<FermionSu2> {
    let rule = Arc::new(FermionParityFusionRule.product(SU2FusionRule));
    GradedSpace::try_new_with_arc(
        rule,
        [
            (
                product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(0)),
                2,
            ),
            (
                product_sector(Z2Irrep::ODD, SU2Irrep::from_twice_spin(2)),
                1,
            ),
        ],
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// Transfer
// ---------------------------------------------------------------------------

fn assert_round_trip_is_bit_exact<R, D>(host: &TensorMap<R, D>)
where
    R: DeviceRule,
    D: DevicePayload,
{
    let device = host.to_cuda().unwrap();
    let back = device.to_host().unwrap();
    assert_bit_exact(back.data(), host.data(), "round trip");
    assert_eq!(back.rank(), host.rank());
    assert_eq!(back.block_count(), host.block_count());
    assert_eq!(back.leg_dims().unwrap(), host.leg_dims().unwrap());

    // A lazy adjoint transfers only its parent and rebuilds a lazy view; the
    // logical payload is read by materializing both sides the same way.
    let one = D::entry(1.0, 0.0);
    let lazy = host.adjoint().unwrap();
    let lazy_back = lazy.to_cuda().unwrap().to_host().unwrap();
    assert_bit_exact(
        lazy_back.scale(one).data(),
        lazy.scale(one).data(),
        "lazy adjoint round trip",
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_transfer_round_trips_bit_exactly_at_every_payload() {
    let runtime = runtime();
    let u1 = u1_leg();
    let su2 = su2_leg();
    let fz2su2 = fz2_su2_leg();

    fn run<R: DeviceRule, D: DevicePayload>(runtime: &Runtime, leg: &GradedSpace<R>) {
        let host = TensorMap::<R, D>::from_block_fn(runtime, [leg, leg], [leg], fill::<D, _>(1.5))
            .unwrap();
        assert_round_trip_is_bit_exact(&host);
    }

    run::<_, f64>(&runtime, &u1);
    run::<_, Complex64>(&runtime, &u1);
    run::<_, f32>(&runtime, &u1);
    run::<_, Complex32>(&runtime, &u1);

    run::<_, f64>(&runtime, &su2);
    run::<_, f32>(&runtime, &su2);
    run::<_, Complex32>(&runtime, &su2);

    run::<_, f32>(&runtime, &fz2su2);
    run::<_, Complex32>(&runtime, &fz2su2);
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

fn assert_arithmetic_matches_host<R, D>(runtime: &Runtime, leg: &GradedSpace<R>)
where
    R: DeviceRule,
    D: DevicePayload,
{
    let a = TensorMap::<R, D>::from_block_fn(runtime, [leg], [leg], fill::<D, _>(1.5)).unwrap();
    let b = TensorMap::<R, D>::from_block_fn(runtime, [leg], [leg], fill::<D, _>(-2.25)).unwrap();
    let alpha = D::entry(2.0, -3.0);
    let beta = D::entry(-0.5, 1.25);
    let terms = a.data().len();
    let tolerance = elementwise_tolerance::<D>(2, 8.0);

    let device_a = a.to_cuda().unwrap();
    let device_b = b.to_cuda().unwrap();

    let scaled = device_a.scale(alpha).unwrap().to_host().unwrap();
    assert_close(scaled.data(), a.scale(alpha).data(), tolerance, "scale");
    assert_moved(a.data(), scaled.data(), "scale");

    let summed = device_a
        .add(&device_b, alpha, beta)
        .unwrap()
        .to_host()
        .unwrap();
    assert_close(
        summed.data(),
        a.add(&b, alpha, beta).unwrap().data(),
        tolerance,
        "add",
    );

    let zeros = device_a.zeros_like().unwrap().to_host().unwrap();
    assert_bit_exact(zeros.data(), &vec![D::entry(0.0, 0.0); terms], "zeros_like");

    // `normalize` divides by a device-accumulated norm, so it carries the
    // reduction bound rather than the elementwise one.
    let normalized = device_a.normalize().unwrap().to_host().unwrap();
    assert_close(
        normalized.data(),
        a.normalize().unwrap().data(),
        reduction_tolerance::<D>(terms, 8.0),
        "normalize",
    );

    // The lazy fold: `alpha A^H + beta B^H` over the device parents.
    let lazy_a = device_a.adjoint().unwrap();
    let lazy_b = device_b.adjoint().unwrap();
    let host_fold = a
        .adjoint()
        .unwrap()
        .add(&b.adjoint().unwrap(), alpha, beta)
        .unwrap();
    assert_close(
        lazy_a
            .add(&lazy_b, alpha, beta)
            .unwrap()
            .to_host()
            .unwrap()
            .data(),
        host_fold.data(),
        tolerance,
        "lazy adjoint fold",
    );
    assert_close(
        lazy_a.scale(alpha).unwrap().to_host().unwrap().data(),
        a.adjoint().unwrap().scale(alpha).data(),
        tolerance,
        "lazy adjoint scale",
    );

    // Mixing a lazy and an owned operand stays the documented device scope at
    // every dtype: the rejection order is unchanged by the payload.
    assert!(matches!(
        lazy_a.add(&device_b, alpha, beta),
        Err(Error::UnsupportedOnDevice(_))
    ));
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_arithmetic_matches_the_host_at_every_payload() {
    let runtime = runtime();
    let u1 = u1_leg();
    let su2 = su2_leg();

    assert_arithmetic_matches_host::<_, f64>(&runtime, &u1);
    assert_arithmetic_matches_host::<_, Complex64>(&runtime, &u1);
    assert_arithmetic_matches_host::<_, f32>(&runtime, &u1);
    assert_arithmetic_matches_host::<_, Complex32>(&runtime, &u1);
    assert_arithmetic_matches_host::<_, f32>(&runtime, &su2);
    assert_arithmetic_matches_host::<_, Complex32>(&runtime, &su2);
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

fn assert_reductions_match_host<R, D>(runtime: &Runtime, leg: &GradedSpace<R>)
where
    R: DeviceRule,
    D: DevicePayload,
{
    let a = TensorMap::<R, D>::from_block_fn(runtime, [leg], [leg], fill::<D, _>(1.5)).unwrap();
    let b = TensorMap::<R, D>::from_block_fn(runtime, [leg], [leg], fill::<D, _>(-2.25)).unwrap();
    let terms = a.data().len();
    // An upper bound on `sum |conj(a_i) * b_i|`, which is what the device
    // reduction's error bound is relative to.
    let peak = |tensor: &TensorMap<R, D>| {
        tensor
            .data()
            .iter()
            .map(|value| value.magnitude())
            .fold(0.0_f64, f64::max)
    };
    let scale = peak(&a) * peak(&b) * terms as f64;
    let tolerance = reduction_tolerance::<D>(terms, scale);

    let device_a = a.to_cuda().unwrap();
    let device_b = b.to_cuda().unwrap();

    let host_norm = a.norm().unwrap();
    let device_norm = device_a.norm().unwrap();
    // `norm` takes a square root of the reduction, so its error is the
    // reduction's own bound or the rounding of a result of that magnitude,
    // whichever is larger. Both halves are absolute and documented.
    assert!(
        (device_norm - host_norm).abs() <= tolerance.max(host_norm * D::EPS * terms as f64),
        "norm [{}]: device {device_norm}, host {host_norm}",
        D::NAME
    );
    assert!(host_norm > 0.0, "norm [{}] fixture is vacuous", D::NAME);

    assert_close_absolutely(
        &[device_a.inner(&device_b).unwrap()],
        &[a.inner(&b).unwrap()],
        tolerance,
        "inner",
    );
    // `dot` is the deprecated alias; it must still reduce identically.
    #[allow(deprecated)]
    let device_dot = device_a.dot(&device_b).unwrap();
    assert_eq!(
        device_dot,
        device_a.inner(&device_b).unwrap(),
        "dot [{}] must be the alias of inner",
        D::NAME
    );

    // Conjugate linearity in the first argument, which is what distinguishes
    // `inner` from an unconjugated bilinear form for the complex payloads.
    let phase = D::entry(0.0, 1.0);
    if phase != D::entry(0.0, 0.0) {
        let scaled_lhs = device_a.scale(phase).unwrap();
        assert_close_absolutely(
            &[scaled_lhs.inner(&device_b).unwrap()],
            &[a.scale(phase).inner(&b).unwrap()],
            tolerance,
            "inner conjugate linearity",
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_reductions_match_the_host_within_the_device_accumulation_bound() {
    let runtime = runtime();
    let u1 = u1_leg();
    let su2 = su2_leg();

    assert_reductions_match_host::<_, f64>(&runtime, &u1);
    assert_reductions_match_host::<_, Complex64>(&runtime, &u1);
    assert_reductions_match_host::<_, f32>(&runtime, &u1);
    assert_reductions_match_host::<_, Complex32>(&runtime, &u1);
    // SU(2): several coupled sectors carrying different quantum dimensions, so
    // the wide *cross-sector* half of the contract is exercised too.
    assert_reductions_match_host::<_, f32>(&runtime, &su2);
    assert_reductions_match_host::<_, Complex32>(&runtime, &su2);
}

/// The documented single-precision boundary of the device reduction: the
/// within-sector sum runs in the payload dtype on the device, so it saturates
/// where the host's wide accumulator does not. This is a precision boundary,
/// reported as `inf`, not a typed error — exactly as the same overflow is at
/// `f64`.
#[test]
#[ignore = "requires a real CUDA device"]
fn device_single_precision_norm_can_overflow_where_the_host_stays_finite() {
    let runtime = runtime();
    let leg = u1_leg();
    // The fixture is `[leg] <- [leg]` over `u1_leg`, so its three coupled
    // sectors hold 2x2, 1x1 and 2x2 blocks: 9 entries, at most **4** in one
    // sector. Each `|value|^2` is `1e38`, inside `f32` range; the largest
    // per-sector sum is `4e38`, which is not — `f32::MAX` is `3.4028e38`, a
    // margin of about 1.18x. Every term is positive and of equal magnitude,
    // so no summation order or FMA contraction inside the GEMM avoids the
    // saturation. The Host sums all 9 in `f64`, where `9e38` is nine orders
    // short of `f64::MAX`.
    let big = 1.0e19_f64;
    let host =
        TensorMap::<U1FusionRule, f32>::from_block_fn(&runtime, [&leg], [&leg], |_, _| big as f32)
            .unwrap();
    assert!(host.data().iter().all(|value| value.is_finite()));

    let host_norm = host.norm().unwrap();
    assert!(
        host_norm.is_finite(),
        "the host accumulates in f64 and stays finite, got {host_norm}"
    );

    let device_norm = host.to_cuda().unwrap().norm().unwrap();
    assert!(
        device_norm.is_infinite(),
        "the device accumulates in f32 within a sector and saturates, got {device_norm}"
    );

    // The double-precision twin of the same fixture shape stays finite on both
    // sides, which is what shows the difference is the payload dtype and not
    // the fixture.
    let twin = TensorMap::<U1FusionRule, f64>::from_block_fn(&runtime, [&leg], [&leg], |_, _| big)
        .unwrap();
    assert!(twin.norm().unwrap().is_finite());
    assert!(twin.to_cuda().unwrap().norm().unwrap().is_finite());
}

/// What the overflow above *does* to the two public operations that consume a
/// device reduction, carried from the #1336 review.
///
/// `inner(a, a)` is the unrooted reduction, so it saturates first and reports
/// `inf` — the same convention `f64` overflow already has, not a typed error.
/// `normalize` then divides every entry by that `inf` and returns an
/// **all-zero tensor with no error**: the documented silent case, asserted
/// here so it cannot become a quiet `NaN` (which is what a `0/inf` on a zero
/// entry, or an `inf/inf`, would produce) or a typed failure.
///
/// The `f64` twin of the same fixture shape is the control: it neither
/// saturates nor zeroes.
#[test]
#[ignore = "requires a real CUDA device"]
fn device_single_precision_normalize_of_an_overflowed_norm_is_all_zero() {
    let runtime = runtime();
    let leg = u1_leg();
    let big = 1.0e19_f64;

    let host =
        TensorMap::<U1FusionRule, f32>::from_block_fn(&runtime, [&leg], [&leg], |_, _| big as f32)
            .unwrap();
    let device = host.to_cuda().unwrap();

    let self_inner = device.inner(&device).unwrap();
    assert!(
        self_inner.is_infinite() && self_inner.is_sign_positive(),
        "the device sums |x|^2 in f32 and saturates, got {self_inner}"
    );

    let normalized = device.normalize().unwrap().to_host().unwrap();
    assert!(
        !normalized.data().is_empty(),
        "the fixture must carry a payload"
    );
    for (index, &value) in normalized.data().iter().enumerate() {
        assert_eq!(
            value, 0.0_f32,
            "normalize by an overflowed norm must zero entry {index}, got {value}"
        );
    }

    let twin = TensorMap::<U1FusionRule, f64>::from_block_fn(&runtime, [&leg], [&leg], |_, _| big)
        .unwrap()
        .to_cuda()
        .unwrap();
    assert!(twin.inner(&twin).unwrap().is_finite());
    let twin_normalized = twin.normalize().unwrap().to_host().unwrap();
    assert!(
        twin_normalized.data().iter().all(|value| *value != 0.0),
        "the double-precision control must not be zeroed"
    );
}

/// Carried from the #1336 review: `alpha = 0` (and `-0`) on an
/// `*_overwrite_into` transform must **overwrite**, not scale-and-accumulate.
///
/// The finite-poison loop in `assert_transforms_match_host` cannot tell the
/// two apart — `0 * 7 == 0` either way. A `NaN`-poisoned destination can:
/// `0 * NaN` is `NaN`, so any route that reads the destination before writing
/// leaves `NaN` behind, while a true overwrite leaves an exact `+0`.
#[test]
#[ignore = "requires a real CUDA device"]
fn a_zero_scale_overwrite_into_clears_a_nan_poisoned_destination() {
    fn assert_cleared<R, D>(runtime: &Runtime, leg: &GradedSpace<R>)
    where
        R: DeviceRule,
        D: DevicePayload,
    {
        let host = TensorMap::<R, D>::from_block_fn(runtime, [leg, leg], [leg], fill::<D, _>(1.5))
            .unwrap();
        let device = host.to_cuda().unwrap();
        let host_permuted = host.permute(&[1, 2], &[0]).unwrap();
        let poisoned = host_permuted.scale(D::entry(f64::NAN, 0.0));
        assert!(
            !poisoned.data().is_empty()
                && poisoned
                    .data()
                    .iter()
                    .all(|value| value.magnitude().is_nan()),
            "the poison fixture [{}] must be entirely NaN",
            D::NAME
        );

        for alpha in [D::entry(0.0, 0.0), D::entry(-0.0, -0.0)] {
            let mut destination = poisoned.to_cuda().unwrap();
            device
                .permute_overwrite_into(&mut destination, &[1, 2], &[0], alpha)
                .unwrap();
            for (index, &value) in destination.to_host().unwrap().data().iter().enumerate() {
                assert_eq!(
                    value,
                    D::entry(0.0, 0.0),
                    "permute_overwrite_into [{}] at alpha {alpha:?} left entry {index} as \
                     {value:?}",
                    D::NAME
                );
            }
        }
    }

    let runtime = runtime();
    let u1 = u1_leg();
    assert_cleared::<_, f64>(&runtime, &u1);
    assert_cleared::<_, Complex64>(&runtime, &u1);
    assert_cleared::<_, f32>(&runtime, &u1);
    assert_cleared::<_, Complex32>(&runtime, &u1);
}

// ---------------------------------------------------------------------------
// Contraction and compose
// ---------------------------------------------------------------------------

fn assert_contract_and_compose_match_host<R, D>(runtime: &Runtime, leg: &GradedSpace<R>)
where
    R: DeviceRule,
    D: DevicePayload,
{
    let a = TensorMap::<R, D>::from_block_fn(runtime, [leg], [leg], fill::<D, _>(1.5)).unwrap();
    let b = TensorMap::<R, D>::from_block_fn(runtime, [leg], [leg], fill::<D, _>(-2.25)).unwrap();
    let tolerance = elementwise_tolerance::<D>(a.data().len(), 64.0);

    let device_a = a.to_cuda().unwrap();
    let device_b = b.to_cuda().unwrap();

    let host_contract = a.contract(&b, &[1], &[0], &[0, 1]).unwrap();
    assert_close(
        device_a
            .contract(&device_b, &[1], &[0], &[0, 1])
            .unwrap()
            .to_host()
            .unwrap()
            .data(),
        host_contract.data(),
        tolerance,
        "contract",
    );

    let host_compose = a.compose(&b).unwrap();
    assert_close(
        device_a
            .compose(&device_b)
            .unwrap()
            .to_host()
            .unwrap()
            .data(),
        host_compose.data(),
        tolerance,
        "compose",
    );

    // A lazy adjoint operand: `A^H . B` on both sides.
    let host_lazy = a.adjoint().unwrap().compose(&b).unwrap();
    assert_close(
        device_a
            .adjoint()
            .unwrap()
            .compose(&device_b)
            .unwrap()
            .to_host()
            .unwrap()
            .data(),
        host_lazy.data(),
        tolerance,
        "lazy adjoint compose",
    );

    // `contract_overwrite_into` writes the same values the returning form does.
    let mut destination = device_a
        .contract(&device_b, &[1], &[0], &[0, 1])
        .unwrap()
        .zeros_like()
        .unwrap();
    device_a
        .contract_overwrite_into(
            &device_b,
            &mut destination,
            &[1],
            &[0],
            &[0, 1],
            D::entry(1.0, 0.0),
        )
        .unwrap();
    assert_close(
        destination.to_host().unwrap().data(),
        host_contract.data(),
        tolerance,
        "contract_overwrite_into",
    );

    // A second write over the same destination overwrites rather than
    // accumulates, at every dtype.
    device_a
        .contract_overwrite_into(
            &device_b,
            &mut destination,
            &[1],
            &[0],
            &[0, 1],
            D::entry(1.0, 0.0),
        )
        .unwrap();
    assert_close(
        destination.to_host().unwrap().data(),
        host_contract.data(),
        tolerance,
        "contract_overwrite_into is idempotent",
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_contract_and_compose_match_the_host_at_every_payload() {
    let runtime = runtime();
    let u1 = u1_leg();
    let su2 = su2_leg();
    let fz2su2 = fz2_su2_leg();

    assert_contract_and_compose_match_host::<_, f64>(&runtime, &u1);
    assert_contract_and_compose_match_host::<_, Complex64>(&runtime, &u1);
    assert_contract_and_compose_match_host::<_, f32>(&runtime, &u1);
    assert_contract_and_compose_match_host::<_, Complex32>(&runtime, &u1);
    assert_contract_and_compose_match_host::<_, f32>(&runtime, &su2);
    assert_contract_and_compose_match_host::<_, Complex32>(&runtime, &su2);
    assert_contract_and_compose_match_host::<_, f32>(&runtime, &fz2su2);
    assert_contract_and_compose_match_host::<_, Complex32>(&runtime, &fz2su2);
}

/// The hand-computed fermionic sign fixture of the `f64` suite, at every
/// payload: `contract` carries the `-1` the leg crossing produces and
/// `compose` does not. The values are small integers, so this is an exact
/// comparison at every dtype — a sign that the single-precision lowering lost
/// could not hide inside a tolerance.
#[test]
#[ignore = "requires a real CUDA device"]
fn device_fermionic_signs_are_exact_at_every_payload() {
    let runtime = runtime();
    let provider = Arc::new(FermionParityFusionRule);
    let odd = || GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Z2Irrep::ODD, 1)]).unwrap();
    let odd_dual = || odd().try_dual().unwrap();

    fn run<D: DevicePayload>(
        runtime: &Runtime,
        lhs_codomain: &GradedSpace<FermionParityFusionRule>,
        lhs_domain: &GradedSpace<FermionParityFusionRule>,
        rhs_codomain: &GradedSpace<FermionParityFusionRule>,
        rhs_domain: &GradedSpace<FermionParityFusionRule>,
    ) {
        let lhs =
            TensorMap::<_, D>::from_block_fn(runtime, [lhs_codomain], [lhs_domain], |_, _| {
                D::entry(2.0, 0.0)
            })
            .unwrap();
        let rhs =
            TensorMap::<_, D>::from_block_fn(runtime, [rhs_codomain], [rhs_domain], |_, _| {
                D::entry(3.0, 0.0)
            })
            .unwrap();
        let device_lhs = lhs.to_cuda().unwrap();
        let device_rhs = rhs.to_cuda().unwrap();

        assert_bit_exact(
            device_lhs
                .contract(&device_rhs, &[1], &[0], &[0, 1])
                .unwrap()
                .to_host()
                .unwrap()
                .data(),
            &[D::entry(-6.0, 0.0)],
            "fermionic contract",
        );
        assert_bit_exact(
            device_lhs
                .compose(&device_rhs)
                .unwrap()
                .to_host()
                .unwrap()
                .data(),
            &[D::entry(6.0, 0.0)],
            "fermionic compose",
        );
        // The host agrees, at this dtype, with the same hand value.
        assert_bit_exact(
            lhs.contract(&rhs, &[1], &[0], &[0, 1]).unwrap().data(),
            &[D::entry(-6.0, 0.0)],
            "fermionic contract, host",
        );
    }

    run::<f64>(&runtime, &odd(), &odd_dual(), &odd_dual(), &odd());
    run::<Complex64>(&runtime, &odd(), &odd_dual(), &odd_dual(), &odd());
    run::<f32>(&runtime, &odd(), &odd_dual(), &odd_dual(), &odd());
    run::<Complex32>(&runtime, &odd(), &odd_dual(), &odd_dual(), &odd());
}

// ---------------------------------------------------------------------------
// Structural transforms
// ---------------------------------------------------------------------------

fn assert_transforms_match_host<R, D>(runtime: &Runtime, leg: &GradedSpace<R>)
where
    R: DeviceRule,
    D: DevicePayload,
{
    let host =
        TensorMap::<R, D>::from_block_fn(runtime, [leg, leg], [leg], fill::<D, _>(1.5)).unwrap();
    let device = host.to_cuda().unwrap();
    // A recoupling combines at most the tree cardinality of one block; three
    // legs over these fixtures keeps that small, so `16` is generous.
    let tolerance = elementwise_tolerance::<D>(16, 16.0);

    let host_permuted = host.permute(&[1, 2], &[0]).unwrap();
    let device_permuted = device.permute(&[1, 2], &[0]).unwrap().to_host().unwrap();
    assert_close(
        device_permuted.data(),
        host_permuted.data(),
        tolerance,
        "permute",
    );
    assert_moved(host.data(), device_permuted.data(), "permute");

    let levels: Vec<usize> = (0..host.rank()).collect();
    assert_close(
        device
            .braid(&[1, 2], &[0], &levels)
            .unwrap()
            .to_host()
            .unwrap()
            .data(),
        host.braid(&[1, 2], &[0], &levels).unwrap().data(),
        tolerance,
        "braid",
    );

    assert_close(
        device.repartition(1).unwrap().to_host().unwrap().data(),
        host.repartition(1).unwrap().data(),
        tolerance,
        "repartition",
    );

    assert_close(
        device.transpose().unwrap().to_host().unwrap().data(),
        host.transpose().unwrap().data(),
        tolerance,
        "transpose",
    );

    // The `*_overwrite_into` forms of the same four transforms (#1339) write
    // what their returning twins return, at every caller scale.
    // Each destination starts poisoned with a different multiple of the
    // expected payload, so an implementation that wrote nothing at `alpha = 1`
    // could not pass either.
    let poison = D::entry(7.0, -3.0);
    for alpha in [
        D::entry(1.0, 0.0),
        D::entry(-2.5, 0.5),
        D::entry(0.0, 0.0),
        D::entry(-0.0, -0.0),
    ] {
        let mut destination = host_permuted.scale(poison).to_cuda().unwrap();
        device
            .permute_overwrite_into(&mut destination, &[1, 2], &[0], alpha)
            .unwrap();
        assert_close(
            destination.to_host().unwrap().data(),
            host_permuted.scale(alpha).data(),
            tolerance,
            "permute_overwrite_into",
        );

        let host_transposed = host.transpose().unwrap();
        let mut transposed = host_transposed.scale(poison).to_cuda().unwrap();
        device
            .transpose_overwrite_into(&mut transposed, alpha)
            .unwrap();
        assert_close(
            transposed.to_host().unwrap().data(),
            host_transposed.scale(alpha).data(),
            tolerance,
            "transpose_overwrite_into",
        );

        let host_bent = host.repartition(1).unwrap();
        let mut bent = host_bent.scale(poison).to_cuda().unwrap();
        device.repartition_overwrite_into(&mut bent, alpha).unwrap();
        assert_close(
            bent.to_host().unwrap().data(),
            host_bent.scale(alpha).data(),
            tolerance,
            "repartition_overwrite_into",
        );

        let host_cyclic = host.transpose_axes(&[2], &[1, 0]).unwrap();
        let mut cyclic = host_cyclic.scale(poison).to_cuda().unwrap();
        device
            .transpose_axes_overwrite_into(&mut cyclic, &[2], &[1, 0], alpha)
            .unwrap();
        assert_close(
            cyclic.to_host().unwrap().data(),
            host_cyclic.scale(alpha).data(),
            tolerance,
            "transpose_axes_overwrite_into",
        );
    }

    // Round trip: permuting back reproduces the source at the payload's own
    // precision.
    assert_close(
        device
            .permute(&[1, 2], &[0])
            .unwrap()
            .permute(&[2, 0], &[1])
            .unwrap()
            .to_host()
            .unwrap()
            .data(),
        host.data(),
        tolerance,
        "permute round trip",
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_transforms_match_the_host_at_every_payload() {
    let runtime = runtime();
    let u1 = u1_leg();
    let su2 = su2_leg();
    let fz2su2 = fz2_su2_leg();

    assert_transforms_match_host::<_, f64>(&runtime, &u1);
    assert_transforms_match_host::<_, f32>(&runtime, &u1);
    assert_transforms_match_host::<_, Complex32>(&runtime, &u1);
    // SU(2): real recoupling coefficients acting on a single-precision payload.
    assert_transforms_match_host::<_, f64>(&runtime, &su2);
    assert_transforms_match_host::<_, f32>(&runtime, &su2);
    assert_transforms_match_host::<_, Complex32>(&runtime, &su2);
    // fZ2 ⊠ SU(2): fermionic signs and recoupling together.
    assert_transforms_match_host::<_, f32>(&runtime, &fz2su2);
    assert_transforms_match_host::<_, Complex32>(&runtime, &fz2su2);
}

/// The warm `*_overwrite_into` contract of #1339 does not depend on the
/// payload dtype: a warm replay over a caller-owned destination transfers
/// nothing and allocates nothing at single precision too. The zero-scale route
/// uploads at most one *element* of the payload's own zero template, once per
/// context — half the bytes of the double-precision template, never a buffer.
#[test]
#[ignore = "requires a real CUDA device"]
fn a_warm_single_precision_overwrite_into_transfers_nothing() {
    let runtime = runtime();
    let leg = u1_leg();

    fn run<D: DevicePayload>(runtime: &Runtime, leg: &GradedSpace<U1FusionRule>) {
        let host = TensorMap::<U1FusionRule, D>::from_block_fn(
            runtime,
            [leg, leg],
            [leg],
            fill::<D, _>(1.5),
        )
        .unwrap();
        let device = host.to_cuda().unwrap();
        let mut destination = host.permute(&[1, 2], &[0]).unwrap().to_cuda().unwrap();

        // Cold: prepares the structure for this pair and this dtype.
        device
            .permute_overwrite_into(&mut destination, &[1, 2], &[0], D::entry(1.0, 0.0))
            .unwrap();
        let cold = runtime.cuda_tree_transform_stats().unwrap();

        let before = cuda_transfer_stats();
        device
            .permute_overwrite_into(&mut destination, &[1, 2], &[0], D::entry(-2.5, 0.5))
            .unwrap();
        let after = cuda_transfer_stats();
        assert_eq!(
            (
                after.h2d_calls - before.h2d_calls,
                after.h2d_bytes - before.h2d_bytes,
                after.d2h_calls - before.d2h_calls,
                after.device_allocs - before.device_allocs,
            ),
            (0, 0, 0, 0),
            "warm overwrite [{}] must transfer and allocate nothing",
            D::NAME
        );
        assert_eq!(
            runtime.cuda_tree_transform_stats().unwrap(),
            cold,
            "a warm overwrite [{}] grows no device state",
            D::NAME
        );

        // The zero-scale route sizes the context zero template once, to one
        // element of *this* payload.
        let before = cuda_transfer_stats();
        device
            .permute_overwrite_into(&mut destination, &[1, 2], &[0], D::entry(0.0, 0.0))
            .unwrap();
        let after = cuda_transfer_stats();
        assert!(
            after.h2d_calls - before.h2d_calls <= 1
                && after.h2d_bytes - before.h2d_bytes <= std::mem::size_of::<D>() as u64,
            "the zero template [{}] is one element, not a buffer",
            D::NAME
        );

        let before = cuda_transfer_stats();
        device
            .permute_overwrite_into(&mut destination, &[1, 2], &[0], D::entry(-0.0, -0.0))
            .unwrap();
        let after = cuda_transfer_stats();
        assert_eq!(
            (
                after.h2d_calls - before.h2d_calls,
                after.d2h_calls - before.d2h_calls,
                after.device_allocs - before.device_allocs,
            ),
            (0, 0, 0),
            "a warm zero-scale overwrite [{}] transfers nothing",
            D::NAME
        );
    }

    run::<f64>(&runtime, &leg);
    run::<f32>(&runtime, &leg);
    run::<Complex32>(&runtime, &leg);
}

// ---------------------------------------------------------------------------
// Cost contracts
// ---------------------------------------------------------------------------

/// Same device work, half the bytes.
///
/// The comparison is relative and same-process: the `f32` run of a fixture is
/// compared against the `f64` run of the *same* fixture in the same test
/// binary, so no absolute constant of a particular platform appears. Call
/// counts must be equal (the schedule does not depend on the payload dtype)
/// and transferred bytes must be exactly half (the element size does).
#[test]
#[ignore = "requires a real CUDA device"]
fn single_precision_costs_the_same_device_calls_and_half_the_bytes() {
    let runtime = runtime();
    let leg = u1_leg();

    fn measure<D: DevicePayload>(
        runtime: &Runtime,
        leg: &GradedSpace<U1FusionRule>,
    ) -> (u64, u64, u64, u64, u64, u64) {
        let a =
            TensorMap::<U1FusionRule, D>::from_block_fn(runtime, [leg], [leg], fill::<D, _>(1.5))
                .unwrap();
        let b =
            TensorMap::<U1FusionRule, D>::from_block_fn(runtime, [leg], [leg], fill::<D, _>(-2.25))
                .unwrap();
        let before = cuda_transfer_stats();
        let device_a = a.to_cuda().unwrap();
        let device_b = b.to_cuda().unwrap();
        let product = device_a.contract(&device_b, &[1], &[0], &[0, 1]).unwrap();
        let _ = product.to_host().unwrap();
        let after = cuda_transfer_stats();
        (
            after.h2d_calls - before.h2d_calls,
            after.h2d_bytes - before.h2d_bytes,
            after.d2h_calls - before.d2h_calls,
            after.d2h_bytes - before.d2h_bytes,
            after.device_allocs - before.device_allocs,
            after.gemm_calls - before.gemm_calls,
        )
    }

    // Warm both lanes first: the first call of a dtype builds its execution
    // lane and its device scalar operands, which are one-time costs and not
    // part of the contract under test.
    let _ = measure::<f64>(&runtime, &leg);
    let _ = measure::<f32>(&runtime, &leg);
    let _ = measure::<Complex64>(&runtime, &leg);
    let _ = measure::<Complex32>(&runtime, &leg);

    for (single, double, what) in [
        (
            measure::<f32>(&runtime, &leg),
            measure::<f64>(&runtime, &leg),
            "f32 against f64",
        ),
        (
            measure::<Complex32>(&runtime, &leg),
            measure::<Complex64>(&runtime, &leg),
            "c32 against c64",
        ),
    ] {
        assert_eq!(
            (single.0, single.2, single.4, single.5),
            (double.0, double.2, double.4, double.5),
            "{what}: (h2d_calls, d2h_calls, device_allocs, gemm_calls) must be equal"
        );
        eprintln!("{what}: single {single:?} double {double:?}");
        assert!(double.1 > 0 && double.3 > 0, "{what}: vacuous byte counts");
        assert_eq!(single.1 * 2, double.1, "{what}: h2d bytes must be halved");
        assert_eq!(single.3 * 2, double.3, "{what}: d2h bytes must be halved");
    }
}

// ---------------------------------------------------------------------------
// Device-free gates (no `#[ignore]`: a `cuda` build runs them anywhere)
// ---------------------------------------------------------------------------

/// The admission table itself, as a compile-time fact: the base and
/// factorization device families are open for all four payloads, and device QR
/// for the two real ones. The negative half — a complex device QR, and a
/// generic body that names only the wider marker — is pinned by the
/// `compile_fail` doctests; this is their positive twin.
#[test]
fn the_device_admission_markers_hold_exactly_where_the_table_says() {
    fn device_payload<D: tenet::typed::CudaPayload>() {}
    fn device_factorization_payload<D: tenet::typed::CudaFactorizationPayload>() {}
    fn device_qr_payload<D: tenet::typed::CudaQrPayload>() {}

    device_payload::<f64>();
    device_payload::<Complex64>();
    device_payload::<f32>();
    device_payload::<Complex32>();

    device_factorization_payload::<f64>();
    device_factorization_payload::<Complex64>();
    device_factorization_payload::<f32>();
    device_factorization_payload::<Complex32>();

    device_qr_payload::<f64>();
    device_qr_payload::<f32>();
}

/// The `CudaQrPayload` membership is a projection of the adapter capability
/// constant, and the constant is the authority.
///
/// The static assertion beside the marker enforces the same equality at
/// compile time; this is the runtime statement of it, so the relationship is
/// visible from the test suite and not only from a `const` block.
#[test]
fn device_qr_admission_follows_the_adapter_capability_constant() {
    use tenet::dense::CudaScalar;

    for (name, has_kernels, admitted) in [
        ("f64", <f64 as CudaScalar>::DEVICE_CONSTANT_KERNELS, true),
        ("f32", <f32 as CudaScalar>::DEVICE_CONSTANT_KERNELS, true),
        (
            "c64",
            <Complex64 as CudaScalar>::DEVICE_CONSTANT_KERNELS,
            false,
        ),
        (
            "c32",
            <Complex32 as CudaScalar>::DEVICE_CONSTANT_KERNELS,
            false,
        ),
    ] {
        assert_eq!(
            has_kernels, admitted,
            "{name}: device QR admission and DEVICE_CONSTANT_KERNELS disagree"
        );
    }
}

/// The rejection order does not depend on the payload dtype: a runtime built
/// without a CUDA device rejects an upload with the same typed error at every
/// payload, before any device work.
#[test]
fn a_device_less_runtime_rejects_every_payload_the_same_way() {
    let runtime = Runtime::builder().build().unwrap();
    let leg = u1_leg();

    fn reject<D: DevicePayload>(runtime: &Runtime, leg: &GradedSpace<U1FusionRule>) -> String {
        let host =
            TensorMap::<U1FusionRule, D>::from_block_fn(runtime, [leg], [leg], fill::<D, _>(1.5))
                .unwrap();
        match host.to_cuda() {
            Ok(_) => panic!("a device-less runtime must not upload [{}]", D::NAME),
            Err(error) => format!("{error}"),
        }
    }

    let double = reject::<f64>(&runtime, &leg);
    assert_eq!(reject::<f32>(&runtime, &leg), double);
    assert_eq!(reject::<Complex32>(&runtime, &leg), double);
    assert_eq!(reject::<Complex64>(&runtime, &leg), double);
}
