//! Device tests for the tree-transform executor (issues #1304, #1310).
//!
//! Oracles: the explicit-index walk in `common`, which never reads a
//! `TreeTransformStructure` and is itself pinned against the host executor by
//! `tree_transform_device_oracle.rs` (CPU-only, runs in CI). Every value test
//! additionally compares against the host executor replaying the *same*
//! compiled structure, which catches a device/host divergence even where both
//! disagree with the fixture's intent.
//!
//! The device is the unit under test, so every test is `#[ignore]` like the
//! rest of the device suite.

#![cfg(feature = "cuda")]

mod common;

use std::sync::Mutex;

use common::{
    all_fixtures, expert_interleaved_destination, expert_interleaved_recoupling_destination,
    inactive_destination_layouts, many_distinct_signatures, mixed_single_and_multi, rank_sweep,
    recoupling_non_symmetric_u, unit_coefficient_fixtures, Fixture, TestScalar,
};
use num_complex::Complex64;
use tenet_dense::{
    cuda_transfer_stats, reset_cuda_transfer_stats, CudaDenseContext, CudaScalar, CudaTransferStats,
};
use tenet_operations::cuda::CudaStorage;
use tenet_operations::{
    tree_transform_structure_overwrite_with_strided_kernel_raw,
    tree_transform_structure_with_strided_kernel_raw, CudaTreeTransformDestination,
    CudaTreeTransformExecutor, OperationError, StridedHostKernelAdapter, TreeTransformWorkspace,
    DEFAULT_PLAN_CACHE_BUDGET_BYTES,
};

/// The boundary counters are process-wide, so tests that assert on their
/// deltas must not overlap.
static COUNTER_TESTS: Mutex<()> = Mutex::new(());

/// Payload dtypes replayed on device, with the host arithmetic the oracle and
/// the host comparison need.
trait DeviceScalar:
    TestScalar
    + CudaScalar
    + tenet_operations::TreeTransformScalar
    + tenet_operations::RecouplingCoefficientAction<f64>
    + tenet_operations::DenseBlockScalar
{
}

impl DeviceScalar for f64 {}
impl DeviceScalar for Complex64 {}

fn context() -> CudaDenseContext {
    CudaDenseContext::new(0).expect("a CUDA device")
}

fn host_replay<T: DeviceScalar>(
    fixture: &Fixture,
    source: &[T],
    destination: &[T],
    overwrite: bool,
) -> Vec<T> {
    let structure = fixture.compile();
    let mut kernels = StridedHostKernelAdapter::default();
    let mut workspace = TreeTransformWorkspace::<T>::default();
    let mut data = destination.to_vec();
    let one = T::from_parts(1.0, 0.0);
    if overwrite {
        tree_transform_structure_overwrite_with_strided_kernel_raw(
            &mut kernels,
            &mut workspace,
            &structure,
            &fixture.dst_structure(),
            &fixture.src_structure(),
            &mut data,
            source,
            one,
        )
        .unwrap();
    } else {
        tree_transform_structure_with_strided_kernel_raw(
            &mut kernels,
            &mut workspace,
            &structure,
            &fixture.dst_structure(),
            &fixture.src_structure(),
            &mut data,
            source,
            one,
            one,
        )
        .unwrap();
    }
    data
}

/// Replays `fixture` on device and downloads the destination.
fn device_replay<T: DeviceScalar>(
    ctx: &mut CudaDenseContext,
    executor: &mut CudaTreeTransformExecutor,
    fixture: &Fixture,
    source: &[T],
    destination: &[T],
    overwrite: bool,
) -> Vec<T> {
    let structure = fixture.compile();
    let mut device_dst = CudaStorage::<T>::upload(ctx, destination).unwrap();
    let device_src = CudaStorage::<T>::upload(ctx, source).unwrap();
    let mode = if overwrite {
        CudaTreeTransformDestination::Overwrite
    } else {
        CudaTreeTransformDestination::Axpby(T::from_parts(1.0, 0.0))
    };
    executor
        .replay(
            ctx,
            &structure,
            &fixture.dst_structure(),
            &fixture.src_structure(),
            &mut device_dst,
            &device_src,
            mode,
        )
        .unwrap();
    device_dst.download(ctx).unwrap()
}

fn assert_close<T: TestScalar>(actual: &[T], expected: &[T], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (left, right)) in actual.iter().zip(expected).enumerate() {
        assert!(
            left.distance(*right) <= 1e-12,
            "{what}: element {index} is {left:?}, expected {right:?}"
        );
    }
}

fn check_fixture<T: DeviceScalar>(
    ctx: &mut CudaDenseContext,
    executor: &mut CudaTreeTransformExecutor,
    fixture: &Fixture,
) {
    let source = fixture.source::<T>();
    let destination: Vec<T> = (0..fixture.dst_len())
        .map(|index| T::from_parts(-3.0 - index as f64, 0.5))
        .collect();
    for overwrite in [true, false] {
        let what = format!("{} / {} / overwrite = {overwrite}", fixture.name, T::NAME);
        let device = device_replay(ctx, executor, fixture, &source, &destination, overwrite);
        assert_close(
            &device,
            &fixture.expected(&source, &destination, overwrite),
            &format!("{what}: device vs oracle"),
        );
        assert_close(
            &device,
            &host_replay(fixture, &source, &destination, overwrite),
            &format!("{what}: device vs host"),
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_replay_matches_the_oracle_and_the_host_for_every_fixture() {
    // What: rank 2-6 permutes, transposes, fermionic signs, coefficients that
    // are neither 1 nor -1, conjugated sources, interleaved multi-block
    // layouts, inactive layouts and a zero-extent block, in both payload
    // dtypes and both destination modes.
    let mut ctx = context();
    let mut executor = CudaTreeTransformExecutor::default();
    for fixture in all_fixtures() {
        check_fixture::<f64>(&mut ctx, &mut executor, &fixture);
        check_fixture::<Complex64>(&mut ctx, &mut executor, &fixture);
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn overwrite_cleans_a_nan_poisoned_destination_including_inactive_layouts() {
    // What: the Overwrite mode is destination-independent — every inactive
    // destination layout is zeroed, so a caller's poisoned buffer cannot leak
    // into a block the transform does not reach.
    let mut ctx = context();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = inactive_destination_layouts();
    let source = fixture.source::<f64>();
    let poisoned = vec![f64::NAN; fixture.dst_len()];

    let device = device_replay(&mut ctx, &mut executor, &fixture, &source, &poisoned, true);

    assert!(
        device.iter().all(|value| !value.is_nan()),
        "overwrite left NaN behind: {device:?}"
    );
    assert_close(
        &device,
        &fixture.expected(&source, &poisoned, true),
        "poisoned destination",
    );

    // Negative control: accumulation does not promise a clean destination, so
    // the NaN must survive it — otherwise the test above proves nothing about
    // the zeroing of inactive layouts.
    let accumulated = device_replay(&mut ctx, &mut executor, &fixture, &source, &poisoned, false);
    assert!(
        accumulated.iter().any(|value| value.is_nan()),
        "accumulation must keep the destination's NaN: {accumulated:?}"
    );
}

fn stats_delta(before: CudaTransferStats, after: CudaTransferStats) -> CudaTransferStats {
    CudaTransferStats {
        h2d_calls: after.h2d_calls - before.h2d_calls,
        h2d_bytes: after.h2d_bytes - before.h2d_bytes,
        d2h_calls: after.d2h_calls - before.d2h_calls,
        d2h_bytes: after.d2h_bytes - before.d2h_bytes,
        device_allocs: after.device_allocs - before.device_allocs,
        gemm_calls: after.gemm_calls - before.gemm_calls,
        solver_calls: after.solver_calls - before.solver_calls,
        copy_calls: after.copy_calls - before.copy_calls,
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_warm_replay_transfers_nothing_and_allocates_no_device_buffer() {
    // What: after the first replay of a structure, replaying it again moves
    // nothing across the host boundary and creates no device buffer; only the
    // per-block submissions remain.
    let _guard = COUNTER_TESTS.lock().unwrap();
    let mut ctx = context();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = inactive_destination_layouts();
    let structure = fixture.compile();
    let source = fixture.source::<f64>();
    let destination = vec![0.0_f64; fixture.dst_len()];
    let mut device_dst = CudaStorage::<f64>::upload(&ctx, &destination).unwrap();
    let device_src = CudaStorage::<f64>::upload(&ctx, &source).unwrap();
    let replay = |ctx: &mut CudaDenseContext,
                  executor: &mut CudaTreeTransformExecutor,
                  dst: &mut CudaStorage<f64>| {
        executor
            .replay(
                ctx,
                &structure,
                &fixture.dst_structure(),
                &fixture.src_structure(),
                dst,
                &device_src,
                CudaTreeTransformDestination::Overwrite,
            )
            .unwrap();
    };

    // Cold: the coefficient vector and the zero template are uploaded here.
    replay(&mut ctx, &mut executor, &mut device_dst);
    let cold = cuda_transfer_stats();
    replay(&mut ctx, &mut executor, &mut device_dst);
    let warm = stats_delta(cold, cuda_transfer_stats());

    assert_eq!(warm.h2d_calls, 0, "warm replay uploaded: {warm:?}");
    assert_eq!(warm.d2h_calls, 0, "warm replay downloaded: {warm:?}");
    assert_eq!(warm.device_allocs, 0, "warm replay allocated: {warm:?}");
    // One submission per active block plus one per inactive destination layout.
    assert_eq!(warm.gemm_calls, 3, "submissions changed: {warm:?}");

    // Negative control: a *different* structure does upload its own
    // coefficients, so "no upload" above is a property of reuse, not of the
    // counters being dead.
    let other = many_distinct_signatures(3);
    let cold_other = cuda_transfer_stats();
    let _ = device_replay(
        &mut ctx,
        &mut executor,
        &other,
        &other.source::<f64>(),
        &vec![0.0_f64; other.dst_len()],
        true,
    );
    let delta = stats_delta(cold_other, cuda_transfer_stats());
    assert!(
        delta.h2d_calls > 0,
        "a new structure must upload its coefficients: {delta:?}"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn alternating_structures_upload_their_coefficients_exactly_once_each() {
    // What: the coefficient cache holds more than one structure, so a network
    // replay alternating two transforms does not re-upload on every switch.
    let _guard = COUNTER_TESTS.lock().unwrap();
    let mut ctx = context();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixtures = [rank_sweep().remove(2), interleaved()];
    let prepared: Vec<_> = fixtures
        .iter()
        .map(|fixture| {
            (
                fixture.compile(),
                fixture.dst_structure(),
                fixture.src_structure(),
                fixture.clone(),
            )
        })
        .collect();
    let mut buffers: Vec<_> = prepared
        .iter()
        .map(|(_, _, _, fixture)| {
            let source = fixture.source::<f64>();
            let destination = vec![0.0_f64; fixture.dst_len()];
            (
                CudaStorage::<f64>::upload(&ctx, &destination).unwrap(),
                CudaStorage::<f64>::upload(&ctx, &source).unwrap(),
            )
        })
        .collect();

    let before = cuda_transfer_stats();
    for round in 0..3 {
        for (index, (structure, dst_structure, src_structure, _)) in prepared.iter().enumerate() {
            let (dst, src) = &mut buffers[index];
            executor
                .replay(
                    &mut ctx,
                    structure,
                    dst_structure,
                    src_structure,
                    dst,
                    src,
                    CudaTreeTransformDestination::Overwrite,
                )
                .unwrap();
            assert_eq!(
                executor.prepared_structures(),
                if round == 0 && index == 0 { 1 } else { 2 },
                "round {round}, structure {index}"
            );
        }
    }
    let delta = stats_delta(before, cuda_transfer_stats());

    assert_eq!(
        delta.h2d_calls, 2,
        "exactly one coefficient upload per structure: {delta:?}"
    );
}

fn interleaved() -> Fixture {
    common::interleaved_multi_block()
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_dropped_structure_releases_its_cached_coefficients() {
    // What: the cache keys on a Weak witness of the structure, so device state
    // for a structure the transform cache has evicted cannot survive it.
    let mut ctx = context();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = interleaved();
    let source = fixture.source::<f64>();
    let destination = vec![0.0_f64; fixture.dst_len()];
    let held = many_distinct_signatures(2);

    {
        let doomed = fixture.compile();
        let mut dst = CudaStorage::<f64>::upload(&ctx, &destination).unwrap();
        let src = CudaStorage::<f64>::upload(&ctx, &source).unwrap();
        executor
            .replay(
                &mut ctx,
                &doomed,
                &fixture.dst_structure(),
                &fixture.src_structure(),
                &mut dst,
                &src,
                CudaTreeTransformDestination::Overwrite,
            )
            .unwrap();
        assert_eq!(executor.prepared_structures(), 1);
        assert!(executor.retained_device_bytes(&ctx) > 0);
    }

    // Preparing any other structure purges the dead entry.
    let _ = device_replay(
        &mut ctx,
        &mut executor,
        &held,
        &held.source::<f64>(),
        &vec![0.0_f64; held.dst_len()],
        true,
    );
    assert_eq!(
        executor.prepared_structures(),
        1,
        "the dropped structure's device state must be released"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn more_signatures_than_the_default_plan_bound_raise_the_cap_without_thrashing() {
    // What: a structure with more distinct block layouts than Tenferro's
    // 64-entry cuTENSOR plan bound raises the bound to what it needs, so a warm
    // replay evicts no plan at all.
    let _guard = COUNTER_TESTS.lock().unwrap();
    let mut ctx = context();
    let mut executor = CudaTreeTransformExecutor::new(1 << 20, DEFAULT_PLAN_CACHE_BUDGET_BYTES);
    let fixture = many_distinct_signatures(70);
    let structure = fixture.compile();
    let source = fixture.source::<f64>();
    let destination = vec![0.0_f64; fixture.dst_len()];
    let mut dst = CudaStorage::<f64>::upload(&ctx, &destination).unwrap();
    let src = CudaStorage::<f64>::upload(&ctx, &source).unwrap();
    let default_bound = ctx.plan_cache_max_entries().unwrap();

    let replay = |ctx: &mut CudaDenseContext,
                  executor: &mut CudaTreeTransformExecutor,
                  dst: &mut CudaStorage<f64>| {
        executor
            .replay(
                ctx,
                &structure,
                &fixture.dst_structure(),
                &fixture.src_structure(),
                dst,
                &src,
                CudaTreeTransformDestination::Overwrite,
            )
            .unwrap();
    };
    replay(&mut ctx, &mut executor, &mut dst);

    assert!(
        executor.required_plan_entries() > 64,
        "fixture must exceed the default bound, got {}",
        executor.required_plan_entries()
    );
    assert!(
        ctx.plan_cache_max_entries().unwrap() >= executor.required_plan_entries(),
        "the cap was not raised from {default_bound}"
    );

    let warm_before = ctx.plan_cache_stats().unwrap();
    replay(&mut ctx, &mut executor, &mut dst);
    let warm_after = ctx.plan_cache_stats().unwrap();

    assert_eq!(
        warm_after.evictions, warm_before.evictions,
        "a warm replay must not evict a plan it needs again"
    );
    assert_eq!(
        warm_after.misses, warm_before.misses,
        "every plan of a warm replay is already cached"
    );
    assert_close(
        &dst.download(&ctx).unwrap(),
        &fixture.expected(&source, &destination, true),
        "many signatures",
    );

    // Negative control: an executor whose plan budget buys nothing raises no
    // cap, and the same structure then thrashes the 64-entry default. A fresh
    // context is required because the cap is per backend instance and this
    // test just raised the one above.
    let mut starved_ctx = context();
    let mut starved = CudaTreeTransformExecutor::new(1 << 20, 0);
    let mut starved_dst = CudaStorage::<f64>::upload(&starved_ctx, &destination).unwrap();
    let starved_src = CudaStorage::<f64>::upload(&starved_ctx, &source).unwrap();
    let starved_replay = |ctx: &mut CudaDenseContext,
                          executor: &mut CudaTreeTransformExecutor,
                          dst: &mut CudaStorage<f64>| {
        executor
            .replay(
                ctx,
                &structure,
                &fixture.dst_structure(),
                &fixture.src_structure(),
                dst,
                &starved_src,
                CudaTreeTransformDestination::Overwrite,
            )
            .unwrap();
    };
    starved_replay(&mut starved_ctx, &mut starved, &mut starved_dst);
    assert_eq!(
        starved_ctx.plan_cache_max_entries().unwrap(),
        default_bound,
        "a zero plan budget must raise nothing"
    );
    let starved_before = starved_ctx.plan_cache_stats().unwrap();
    starved_replay(&mut starved_ctx, &mut starved, &mut starved_dst);
    let starved_after = starved_ctx.plan_cache_stats().unwrap();
    assert!(
        starved_after.evictions > starved_before.evictions,
        "without the raise the warm replay must thrash: {starved_before:?} -> {starved_after:?}"
    );
    assert_close(
        &starved_dst.download(&starved_ctx).unwrap(),
        &fixture.expected(&source, &destination, true),
        "many signatures, thrashing plan cache",
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_coefficient_of_one_moves_f64_payloads_bitwise() {
    // What: where the host copies bit-exactly, the device's multiply by the
    // uploaded `1` is exact too, so the two agree to the last bit for finite
    // f64 payloads. (Not a contract for other coefficients or for complex
    // payloads: see the +-inf deviation recorded in G2a-1.)
    let mut ctx = context();
    let mut executor = CudaTreeTransformExecutor::default();
    for fixture in unit_coefficient_fixtures() {
        let source = fixture.source::<f64>();
        let destination = vec![0.0_f64; fixture.dst_len()];
        let device = device_replay(
            &mut ctx,
            &mut executor,
            &fixture,
            &source,
            &destination,
            true,
        );
        assert_eq!(
            device,
            host_replay(&fixture, &source, &destination, true),
            "{} must match the host bitwise",
            fixture.name
        );
        assert_eq!(
            device,
            fixture.expected(&source, &destination, true),
            "{} must match the oracle bitwise",
            fixture.name
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn unsupported_modes_are_rejected_before_any_device_work() {
    // What: a beta and a destination layout the device cannot express are typed
    // capability errors reported before any upload, allocation, submission or
    // plan-cache change — for a Single-block structure and for a recoupling
    // structure, whose pack columns and coefficient upload must not happen on
    // the strength of one unwritable scatter region.
    let _guard = COUNTER_TESTS.lock().unwrap();
    let mut ctx = context();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = interleaved();
    let source = fixture.source::<f64>();
    let destination = vec![0.0_f64; fixture.dst_len()];
    let mut dst = CudaStorage::<f64>::upload(&ctx, &destination).unwrap();
    let src = CudaStorage::<f64>::upload(&ctx, &source).unwrap();

    // A structure whose destination layout the host proves injective only
    // through its exact overlap fallback, which the device region primitive
    // cannot express.
    let expert = expert_interleaved_destination();
    let expert_structure = expert.compile();
    let expert_destination: Vec<f64> = (0..expert.dst_len()).map(|i| 100.0 + i as f64).collect();
    let mut expert_dst = CudaStorage::<f64>::upload(&ctx, &expert_destination).unwrap();
    let expert_src = CudaStorage::<f64>::upload(&ctx, &expert.source::<f64>()).unwrap();

    let recoupling = expert_interleaved_recoupling_destination();
    let recoupling_structure = recoupling.compile();
    let recoupling_destination: Vec<f64> = (0..recoupling.dst_len())
        .map(|i| 200.0 + i as f64)
        .collect();
    let mut recoupling_dst = CudaStorage::<f64>::upload(&ctx, &recoupling_destination).unwrap();
    let recoupling_src = CudaStorage::<f64>::upload(&ctx, &recoupling.source::<f64>()).unwrap();

    reset_cuda_transfer_stats();
    let before = cuda_transfer_stats();
    let plans_before = ctx.plan_cache_stats().unwrap();

    for beta in [0.0_f64, 2.0, -1.0] {
        let error = executor
            .replay(
                &mut ctx,
                &fixture.compile(),
                &fixture.dst_structure(),
                &fixture.src_structure(),
                &mut dst,
                &src,
                CudaTreeTransformDestination::Axpby(beta),
            )
            .unwrap_err();
        assert!(
            matches!(error, OperationError::UnsupportedDeviceTreeTransform { .. }),
            "beta {beta} gave {error:?}"
        );
    }
    let layout = executor
        .replay(
            &mut ctx,
            &expert_structure,
            &expert.dst_structure(),
            &expert.src_structure(),
            &mut expert_dst,
            &expert_src,
            CudaTreeTransformDestination::Overwrite,
        )
        .unwrap_err();
    assert!(
        matches!(
            layout,
            OperationError::UnsupportedDeviceTreeTransform { .. }
        ),
        "an inexpressible destination layout gave {layout:?}"
    );
    assert_eq!(
        expert_dst.download(&ctx).unwrap(),
        expert_destination,
        "a rejected layout must leave the caller's destination untouched"
    );

    let scatter = executor
        .replay(
            &mut ctx,
            &recoupling_structure,
            &recoupling.dst_structure(),
            &recoupling.src_structure(),
            &mut recoupling_dst,
            &recoupling_src,
            CudaTreeTransformDestination::Overwrite,
        )
        .unwrap_err();
    assert!(
        matches!(
            scatter,
            OperationError::UnsupportedDeviceTreeTransform { .. }
        ),
        "an inexpressible scatter destination gave {scatter:?}"
    );
    assert_eq!(
        recoupling_dst.download(&ctx).unwrap(),
        recoupling_destination,
        "a rejected recoupling must leave the caller's destination untouched"
    );
    assert_eq!(
        executor.workspace_device_bytes(),
        0,
        "a rejected recoupling must not have grown the workspace"
    );

    let delta = stats_delta(before, cuda_transfer_stats());
    // The two downloads above are the test's own read-backs.
    let read_back = (expert_destination.len() + recoupling_destination.len()) * size_of::<f64>();
    let delta = CudaTransferStats {
        d2h_calls: delta.d2h_calls - 2,
        d2h_bytes: delta.d2h_bytes - read_back as u64,
        ..delta
    };
    assert_eq!(delta, CudaTransferStats::default(), "rejection did work");
    assert_eq!(
        ctx.plan_cache_stats().unwrap(),
        plans_before,
        "rejection touched the plan cache"
    );
    assert_eq!(executor.prepared_structures(), 0);
    assert_eq!(executor.required_plan_entries(), 0);

    // Negative control: the same structure with beta = 1 is accepted, and a
    // recoupling structure whose destinations *are* writable replays, so the
    // rejections above are about the mode and the layout, not about Multi
    // blocks or about the fixtures.
    executor
        .replay(
            &mut ctx,
            &fixture.compile(),
            &fixture.dst_structure(),
            &fixture.src_structure(),
            &mut dst,
            &src,
            CudaTreeTransformDestination::Axpby(1.0),
        )
        .unwrap();
    assert_eq!(executor.prepared_structures(), 1);

    let accepted = recoupling_non_symmetric_u();
    let accepted_source = accepted.source::<f64>();
    let accepted_destination = vec![0.0_f64; accepted.dst_len()];
    let device = device_replay(
        &mut ctx,
        &mut executor,
        &accepted,
        &accepted_source,
        &accepted_destination,
        true,
    );
    assert_close(
        &device,
        &accepted.expected(&accepted_source, &accepted_destination, true),
        "an accepted recoupling structure",
    );
    assert!(executor.workspace_device_bytes() > 0);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn mismatched_structures_and_lengths_are_rejected_by_admission() {
    // Negative control for the admission wiring: Stage A still checks the
    // structures and the exact storage lengths, and the device buffer's
    // placement is the executor's own.
    let mut ctx = context();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = interleaved();
    let other = inactive_destination_layouts();
    let source = fixture.source::<f64>();
    let destination = vec![0.0_f64; fixture.dst_len()];
    let structure = fixture.compile();
    let mut dst = CudaStorage::<f64>::upload(&ctx, &destination).unwrap();
    let src = CudaStorage::<f64>::upload(&ctx, &source).unwrap();
    let mut short =
        CudaStorage::<f64>::upload(&ctx, &destination[..destination.len() - 1]).unwrap();

    let wrong_structure = executor
        .replay(
            &mut ctx,
            &structure,
            &other.dst_structure(),
            &fixture.src_structure(),
            &mut dst,
            &src,
            CudaTreeTransformDestination::Overwrite,
        )
        .unwrap_err();
    assert!(
        matches!(wrong_structure, OperationError::InvalidArgument { .. }),
        "{wrong_structure:?}"
    );

    let wrong_length = executor
        .replay(
            &mut ctx,
            &structure,
            &fixture.dst_structure(),
            &fixture.src_structure(),
            &mut short,
            &src,
            CudaTreeTransformDestination::Overwrite,
        )
        .unwrap_err();
    assert!(
        matches!(wrong_length, OperationError::InvalidArgument { .. }),
        "{wrong_length:?}"
    );

    // The same call with the right structures and lengths succeeds.
    executor
        .replay(
            &mut ctx,
            &structure,
            &fixture.dst_structure(),
            &fixture.src_structure(),
            &mut dst,
            &src,
            CudaTreeTransformDestination::Overwrite,
        )
        .unwrap();
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_second_context_prepares_its_own_device_state() {
    // What: device state is keyed by context, so a buffer of another context is
    // never read back through a cached coefficient vector.
    let mut first = context();
    let mut second = context();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = interleaved();
    let structure = fixture.compile();
    let source = fixture.source::<f64>();
    let destination = vec![0.0_f64; fixture.dst_len()];

    for ctx in [&mut first, &mut second] {
        let mut dst = CudaStorage::<f64>::upload(ctx, &destination).unwrap();
        let src = CudaStorage::<f64>::upload(ctx, &source).unwrap();
        executor
            .replay(
                ctx,
                &structure,
                &fixture.dst_structure(),
                &fixture.src_structure(),
                &mut dst,
                &src,
                CudaTreeTransformDestination::Overwrite,
            )
            .unwrap();
        assert_close(
            &dst.download(ctx).unwrap(),
            &fixture.expected(&source, &destination, true),
            "per-context replay",
        );
    }
    assert_eq!(
        executor.prepared_structures(),
        2,
        "one entry per (structure, dtype, context)"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn both_payload_dtypes_share_one_structure_with_their_own_coefficients() {
    // What: the same compiled structure replays for f64 and Complex64, each
    // with its own converted coefficient vector.
    let mut ctx = context();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = rank_sweep().remove(4); // conjugated, fermionic sign, rank 4
    let structure = fixture.compile();

    for complex in [false, true] {
        if complex {
            let source = fixture.source::<Complex64>();
            let destination = vec![Complex64::new(0.0, 0.0); fixture.dst_len()];
            let mut dst = CudaStorage::<Complex64>::upload(&ctx, &destination).unwrap();
            let src = CudaStorage::<Complex64>::upload(&ctx, &source).unwrap();
            executor
                .replay(
                    &mut ctx,
                    &structure,
                    &fixture.dst_structure(),
                    &fixture.src_structure(),
                    &mut dst,
                    &src,
                    CudaTreeTransformDestination::Overwrite,
                )
                .unwrap();
            assert_close(
                &dst.download(&ctx).unwrap(),
                &fixture.expected(&source, &destination, true),
                "complex payload of a conjugated source",
            );
        } else {
            let source = fixture.source::<f64>();
            let destination = vec![0.0_f64; fixture.dst_len()];
            let mut dst = CudaStorage::<f64>::upload(&ctx, &destination).unwrap();
            let src = CudaStorage::<f64>::upload(&ctx, &source).unwrap();
            executor
                .replay(
                    &mut ctx,
                    &structure,
                    &fixture.dst_structure(),
                    &fixture.src_structure(),
                    &mut dst,
                    &src,
                    CudaTreeTransformDestination::Overwrite,
                )
                .unwrap();
            assert_close(
                &dst.download(&ctx).unwrap(),
                &fixture.expected(&source, &destination, true),
                "real payload",
            );
        }
    }
    assert_eq!(executor.prepared_structures(), 2, "one entry per dtype");
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_warm_recoupling_replay_transfers_nothing_and_reuses_the_workspace() {
    // What: the pack/scatter workspace and the uploaded recoupling matrices are
    // the whole per-structure device state, so replaying a Multi structure a
    // second time moves nothing across the boundary, allocates nothing, and
    // grows the workspace by nothing.
    let _guard = COUNTER_TESTS.lock().unwrap();
    let mut ctx = context();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = mixed_single_and_multi();
    let structure = fixture.compile();
    let source = fixture.source::<f64>();
    let destination = vec![0.0_f64; fixture.dst_len()];
    let mut device_dst = CudaStorage::<f64>::upload(&ctx, &destination).unwrap();
    let device_src = CudaStorage::<f64>::upload(&ctx, &source).unwrap();
    let replay = |ctx: &mut CudaDenseContext,
                  executor: &mut CudaTreeTransformExecutor,
                  dst: &mut CudaStorage<f64>| {
        executor
            .replay(
                ctx,
                &structure,
                &fixture.dst_structure(),
                &fixture.src_structure(),
                dst,
                &device_src,
                CudaTreeTransformDestination::Overwrite,
            )
            .unwrap();
    };

    replay(&mut ctx, &mut executor, &mut device_dst);
    let cold = cuda_transfer_stats();
    let cold_workspace = executor.workspace_device_bytes();
    assert!(cold_workspace > 0, "a Multi structure needs a workspace");
    replay(&mut ctx, &mut executor, &mut device_dst);
    let warm = stats_delta(cold, cuda_transfer_stats());

    assert_eq!(warm.h2d_calls, 0, "warm replay uploaded: {warm:?}");
    assert_eq!(warm.d2h_calls, 0, "warm replay downloaded: {warm:?}");
    assert_eq!(warm.device_allocs, 0, "warm replay allocated: {warm:?}");
    assert_eq!(
        executor.workspace_device_bytes(),
        cold_workspace,
        "warm replay grew the workspace"
    );
    // Two GEMM jobs, four packs, four scatters, one Single block and one
    // inactive destination layout: every one of them is a submission.
    assert_eq!(warm.gemm_calls, 12, "submissions changed: {warm:?}");
    assert_close(
        &device_dst.download(&ctx).unwrap(),
        &fixture.expected(&source, &destination, true),
        "warm recoupling replay",
    );

    // Negative control: a wider recoupling structure does grow the workspace
    // and does upload, so the equalities above are reuse, not dead counters.
    let wider = many_distinct_recoupling_columns(8);
    let before = cuda_transfer_stats();
    let _ = device_replay(
        &mut ctx,
        &mut executor,
        &wider,
        &wider.source::<f64>(),
        &vec![0.0_f64; wider.dst_len()],
        true,
    );
    let delta = stats_delta(before, cuda_transfer_stats());
    assert!(
        delta.h2d_calls > 0,
        "a new structure must upload: {delta:?}"
    );
    assert!(
        executor.workspace_device_bytes() > cold_workspace,
        "a wider structure must grow the workspace"
    );

    // And replaying the narrow structure again neither shrinks the workspace
    // nor re-uploads: growth is monotonic and the buffers are shared.
    let before = cuda_transfer_stats();
    let grown = executor.workspace_device_bytes();
    replay(&mut ctx, &mut executor, &mut device_dst);
    let delta = stats_delta(before, cuda_transfer_stats());
    assert_eq!(delta.h2d_calls, 0, "re-upload after growth: {delta:?}");
    assert_eq!(delta.device_allocs, 0, "re-allocation after growth");
    assert_eq!(executor.workspace_device_bytes(), grown, "workspace shrank");
}

/// A recoupling structure with `columns` sources and `columns` destinations of
/// a larger degeneracy box than [`mixed_single_and_multi`], used as the
/// workspace-growth control.
fn many_distinct_recoupling_columns(columns: usize) -> Fixture {
    let element_count = 12;
    let mut dst_blocks = Vec::with_capacity(columns);
    let mut src_blocks = Vec::with_capacity(columns);
    for index in 0..columns {
        dst_blocks.push(common::Block::packed(vec![3, 4], index * element_count));
        src_blocks.push(common::Block::packed(vec![4, 3], index * element_count));
    }
    let u = (0..columns * columns)
        .map(|index| 0.5 + (index as f64) * 0.25)
        .collect();
    Fixture {
        name: "many_distinct_recoupling_columns",
        rank: 2,
        dst_blocks,
        src_blocks,
        pairs: Vec::new(),
        groups: vec![common::Group {
            dst_blocks: (0..columns).collect(),
            src_blocks: (0..columns).collect(),
            axes: vec![1, 0],
            u,
        }],
        conjugate: false,
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn alternating_recoupling_structures_upload_their_matrices_exactly_once_each() {
    // What: the recoupling matrices are cached with the block coefficients
    // under the same (structure, dtype, context) key, so a replay alternating
    // two Multi structures uploads each structure's matrices once and shares
    // one workspace between them.
    let _guard = COUNTER_TESTS.lock().unwrap();
    let mut ctx = context();
    let mut executor = CudaTreeTransformExecutor::default();
    // Widest first, so the workspace is allocated once and the narrower
    // structure reuses it: growth is monotonic and shared across structures.
    let fixtures = [mixed_single_and_multi(), recoupling_non_symmetric_u()];
    let prepared: Vec<_> = fixtures
        .iter()
        .map(|fixture| (fixture.compile(), fixture.clone()))
        .collect();
    let mut buffers: Vec<_> = prepared
        .iter()
        .map(|(_, fixture)| {
            (
                CudaStorage::<f64>::upload(&ctx, &vec![0.0_f64; fixture.dst_len()]).unwrap(),
                CudaStorage::<f64>::upload(&ctx, &fixture.source::<f64>()).unwrap(),
            )
        })
        .collect();

    let before = cuda_transfer_stats();
    let mut after_first_round = before;
    for round in 0..3 {
        for (index, (structure, fixture)) in prepared.iter().enumerate() {
            let (dst, src) = &mut buffers[index];
            executor
                .replay(
                    &mut ctx,
                    structure,
                    &fixture.dst_structure(),
                    &fixture.src_structure(),
                    dst,
                    src,
                    CudaTreeTransformDestination::Overwrite,
                )
                .unwrap();
        }
        if round == 0 {
            after_first_round = cuda_transfer_stats();
        }
    }
    let cold = stats_delta(before, after_first_round);
    let warm = stats_delta(after_first_round, cuda_transfer_stats());

    assert_eq!(executor.prepared_structures(), 2);
    // Switching structures again uploads nothing at all: each structure's
    // matrices are resident and the workspace is shared.
    assert_eq!(warm.h2d_calls, 0, "a switch re-uploaded: {warm:?}");
    assert_eq!(warm.device_allocs, 0, "a switch allocated: {warm:?}");
    // The first round pays six: one coefficient-and-matrix vector per
    // structure, the two workspace buffers the wider structure allocates and
    // the narrower one reuses, the context's shared `1` that pack and scatter
    // read as their coefficient, and the zero template the wider structure's
    // inactive destination layout needs.
    assert_eq!(cold.h2d_calls, 6, "cold uploads changed: {cold:?}");
    for (index, (_, fixture)) in prepared.iter().enumerate() {
        let (dst, _) = &buffers[index];
        assert_close(
            &dst.download(&ctx).unwrap(),
            &fixture.expected(
                &fixture.source::<f64>(),
                &vec![0.0_f64; fixture.dst_len()],
                true,
            ),
            fixture.name,
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn recoupling_replays_match_the_oracle_in_both_dtypes_and_modes() {
    // What: the Multi path itself — non-symmetric square U, rectangular U, a
    // conjugated source, and Single and Multi blocks in one structure — over
    // both payload dtypes and both destination modes, against the oracle and
    // against the host executor replaying the same compiled structure.
    let mut ctx = context();
    let mut executor = CudaTreeTransformExecutor::default();
    for fixture in common::recoupling_fixtures() {
        check_fixture::<f64>(&mut ctx, &mut executor, &fixture);
        check_fixture::<Complex64>(&mut ctx, &mut executor, &fixture);
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn overwrite_cleans_a_nan_poisoned_destination_around_recoupling_blocks() {
    // What: Overwrite is destination-independent for a Multi structure too —
    // the scatter assigns and the untouched layout beside it is zeroed, so a
    // poisoned buffer cannot leak through a recoupling block.
    let mut ctx = context();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = mixed_single_and_multi();
    let source = fixture.source::<f64>();
    let poisoned = vec![f64::NAN; fixture.dst_len()];

    let device = device_replay(&mut ctx, &mut executor, &fixture, &source, &poisoned, true);

    assert!(
        device.iter().all(|value| !value.is_nan()),
        "overwrite left NaN behind: {device:?}"
    );
    assert_close(
        &device,
        &fixture.expected(&source, &poisoned, true),
        "poisoned destination around recoupling",
    );

    // Negative control: accumulation keeps the NaN, so the cleanliness above is
    // the Overwrite rule and not an artefact of the fixture.
    let accumulated = device_replay(&mut ctx, &mut executor, &fixture, &source, &poisoned, false);
    assert!(
        accumulated.iter().any(|value| value.is_nan()),
        "accumulation must keep the destination's NaN: {accumulated:?}"
    );
}
