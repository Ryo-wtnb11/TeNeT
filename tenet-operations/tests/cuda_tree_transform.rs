//! Device tests for the Single-block tree-transform executor (issue #1304).
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

use std::sync::{Arc, Mutex};

use common::{
    all_fixtures, inactive_destination_layouts, many_distinct_signatures, rank_sweep, Fixture,
    TestScalar,
};
use num_complex::Complex64;
use tenet_dense::{
    cuda_transfer_stats, reset_cuda_transfer_stats, CudaDenseContext, CudaScalar, CudaTransferStats,
};
use tenet_operations::cuda::CudaStorage;
use tenet_operations::{
    tree_transform_structure_overwrite_with_strided_kernel_raw,
    tree_transform_structure_with_strided_kernel_raw, CudaTreeTransformDestination,
    CudaTreeTransformExecutor, OperationError, StridedHostKernelAdapter, TreeTransformBlockSpec,
    TreeTransformStructure, TreeTransformWorkspace, DEFAULT_PLAN_CACHE_BUDGET_BYTES,
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
}

#[test]
#[ignore = "requires a real CUDA device"]
fn unsupported_modes_are_rejected_before_any_device_work() {
    // What: a recoupling block and a beta the device cannot express are typed
    // capability errors reported before any upload, allocation, submission or
    // plan-cache change.
    let _guard = COUNTER_TESTS.lock().unwrap();
    let mut ctx = context();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = interleaved();
    let source = fixture.source::<f64>();
    let destination = vec![0.0_f64; fixture.dst_len()];
    let mut dst = CudaStorage::<f64>::upload(&ctx, &destination).unwrap();
    let src = CudaStorage::<f64>::upload(&ctx, &source).unwrap();

    // A structure whose single block is a recoupling GEMM (leaf G2a-3).
    let space =
        Arc::new(tenet_core::BlockStructure::packed_column_major(1, [vec![2], vec![2]]).unwrap());
    let multi = TreeTransformStructure::compile_structures(
        &space,
        &space,
        &[TreeTransformBlockSpec::multi(
            vec![0, 1],
            vec![0, 1],
            vec![1.0_f64, 0.0, 0.0, 1.0],
        )],
    )
    .unwrap();
    let mut multi_dst = CudaStorage::<f64>::upload(&ctx, &[0.0_f64; 4]).unwrap();
    let multi_src = CudaStorage::<f64>::upload(&ctx, &[1.0_f64; 4]).unwrap();

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
    let error = executor
        .replay(
            &mut ctx,
            &multi,
            &space,
            &space,
            &mut multi_dst,
            &multi_src,
            CudaTreeTransformDestination::Overwrite,
        )
        .unwrap_err();
    assert!(
        matches!(error, OperationError::UnsupportedDeviceTreeTransform { .. }),
        "a recoupling structure gave {error:?}"
    );

    let delta = stats_delta(before, cuda_transfer_stats());
    assert_eq!(delta, CudaTransferStats::default(), "rejection did work");
    assert_eq!(
        ctx.plan_cache_stats().unwrap(),
        plans_before,
        "rejection touched the plan cache"
    );
    assert_eq!(executor.prepared_structures(), 0);
    assert_eq!(executor.required_plan_entries(), 0);

    // Negative control: the same structure with beta = 1 is accepted, so the
    // rejections above are about the mode, not about the fixture.
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
