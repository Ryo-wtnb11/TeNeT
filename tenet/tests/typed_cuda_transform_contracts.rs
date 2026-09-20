//! Transfer, allocation, statistics and rejection contracts of the typed
//! device structural transforms (issue #1322, G2b-2).
//!
//! These read the process-wide `tenet::dense::cuda_transfer_stats` counters, so
//! they live in their own test binary and run single-threaded: another test
//! submitting device work in the same process would perturb every delta.
//!
//! Every test here needs a real device. The Host half of the same contract —
//! a device-less Runtime reporting no device state and still clearing its
//! store — is in the ungated `typed_transform_host_side.rs`.
//!
//! Run with `cargo test -p tenet-rs --features cuda,cpu-faer --test \
//! typed_cuda_transform_contracts -- --ignored --test-threads=1`.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use tenet::core::{U1FusionRule, U1Irrep};
use tenet::dense::{cuda_transfer_stats, CudaTransferStats};
use tenet::typed::{GradedSpace, Runtime, TensorMap};

fn leg() -> GradedSpace<U1FusionRule> {
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

fn fixture(runtime: &Runtime) -> TensorMap<U1FusionRule, f64> {
    let v = leg();
    TensorMap::from_block_fn(runtime, [&v, &v], [&v, &v], |_, indices| {
        indices.iter().map(|&i| i as f64 + 1.0).sum::<f64>()
    })
    .unwrap()
}

/// Counter deltas across one closure.
fn delta<T>(body: impl FnOnce() -> T) -> (T, CudaTransferStats) {
    let before = cuda_transfer_stats();
    let value = body();
    let after = cuda_transfer_stats();
    (
        value,
        CudaTransferStats {
            h2d_calls: after.h2d_calls - before.h2d_calls,
            h2d_bytes: after.h2d_bytes - before.h2d_bytes,
            d2h_calls: after.d2h_calls - before.d2h_calls,
            d2h_bytes: after.d2h_bytes - before.d2h_bytes,
            device_allocs: after.device_allocs - before.device_allocs,
            gemm_calls: after.gemm_calls - before.gemm_calls,
            solver_calls: after.solver_calls - before.solver_calls,
            copy_calls: after.copy_calls - before.copy_calls,
        },
    )
}

// ---------------------------------------------------------------------------
// Device contracts
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires a real CUDA device"]
fn a_warm_device_transform_uploads_only_its_output_and_downloads_nothing() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let device = fixture(&runtime).to_cuda().unwrap();
    // Sized on the Host, so the device's first call below really is its cold
    // one: measuring the output on device would have prepared the structure.
    let output_bytes =
        std::mem::size_of_val(fixture(&runtime).permute(&[2, 0], &[1, 3]).unwrap().data()) as u64;

    // Cold: the output plus exactly one coefficient payload for the structure
    // — a Single-block permute needs no pack/scatter workspace, so those two
    // uploads are the whole of it.
    let cold_stats_before = runtime.cuda_tree_transform_stats().unwrap();
    let (_, cold) = delta(|| device.permute(&[2, 0], &[1, 3]).unwrap());
    let cold_stats_after = runtime.cuda_tree_transform_stats().unwrap();
    assert_eq!(cold.h2d_calls, 2, "{cold:?}");
    assert_eq!(cold.device_allocs, 2, "{cold:?}");
    assert_eq!(cold_stats_after.workspace_bytes, 0, "{cold_stats_after:?}");
    assert_eq!(cold.d2h_calls, 0, "no device transform downloads: {cold:?}");
    assert_eq!(
        cold_stats_after.prepared_structures,
        cold_stats_before.prepared_structures + 1,
        "exactly one device entry is prepared per structure"
    );

    // Warm: exactly one H2D of the output bytes, one device allocation, no
    // download. This is the #740 constant and nothing else.
    let (_, warm) = delta(|| device.permute(&[2, 0], &[1, 3]).unwrap());
    assert_eq!(warm.h2d_calls, 1, "{warm:?}");
    assert_eq!(warm.h2d_bytes, output_bytes, "{warm:?}");
    assert_eq!(warm.d2h_calls, 0, "{warm:?}");
    assert_eq!(warm.d2h_bytes, 0, "{warm:?}");
    assert_eq!(warm.device_allocs, 1, "{warm:?}");
    let warm_stats = runtime.cuda_tree_transform_stats().unwrap();
    assert_eq!(
        warm_stats.prepared_structures, cold_stats_after.prepared_structures,
        "a warm replay prepares nothing new"
    );
    assert_eq!(
        warm_stats.executor_bytes, cold_stats_after.executor_bytes,
        "a warm replay grows no device state"
    );
    assert!(
        warm_stats.required_plan_entries > 0,
        "the prepared structure declares its plan signatures"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn clearing_the_transform_cache_releases_the_device_executor_state() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let device = fixture(&runtime).to_cuda().unwrap();
    let _ = device.permute(&[2, 0], &[1, 3]).unwrap();
    let _ = device.transpose().unwrap();
    let before = runtime.cuda_tree_transform_stats().unwrap();
    assert!(before.prepared_structures >= 2, "{before:?}");
    assert!(before.executor_bytes > 0, "{before:?}");

    runtime.clear_tree_transform_cache();

    let after = runtime.cuda_tree_transform_stats().unwrap();
    assert_eq!(after.prepared_structures, 0, "{after:?}");
    assert_eq!(after.executor_bytes, 0, "{after:?}");
    assert_eq!(after.workspace_bytes, 0, "{after:?}");
    assert_eq!(after.required_plan_entries, 0, "{after:?}");
    // The shared context operands belong to the context, not the executor, and
    // are reported separately — clearing the executor does not release them.
    assert_eq!(
        after.context_scalar_operand_bytes, before.context_scalar_operand_bytes,
        "context operands are counted once and are not the executor's to drop"
    );
    assert_eq!(runtime.tree_transform_cache_info().entries(), 0);

    // Re-preparing after the clear still produces the same answer.
    let expected = fixture(&runtime).permute(&[2, 0], &[1, 3]).unwrap();
    let again = device.permute(&[2, 0], &[1, 3]).unwrap().to_host().unwrap();
    assert_eq!(again.data(), expected.data());
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_transform_short_circuits_do_no_device_work() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let device = fixture(&runtime).to_cuda().unwrap();
    let expected = fixture(&runtime).data().to_vec();

    let (results, counters) = delta(|| {
        [
            // Identity axis lists, the same split, and a rank-0 transpose all
            // return a clone of the receiver.
            device.permute(&[0, 1], &[2, 3]).unwrap(),
            device.transpose_axes(&[0, 1], &[2, 3]).unwrap(),
            device.braid(&[0, 1], &[2, 3], &[1, 2, 3, 4]).unwrap(),
            device.repartition(2).unwrap(),
        ]
    });
    assert_eq!(
        counters,
        CudaTransferStats::default(),
        "a short-circuited transform must submit nothing: {counters:?}"
    );
    for result in results {
        assert_eq!(result.to_host().unwrap().data(), expected);
    }

    // A rank-0 tensor has no legs to build from, so it comes from a full Host
    // trace (device `trace` is a separate leaf) and is then uploaded.
    let v = leg();
    let endomorphism: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&v], [&v], |_, indices| indices[0] as f64 + 1.0)
            .unwrap();
    let rank_zero = endomorphism.trace_pairs(&[(0, 1)]).unwrap();
    assert_eq!(rank_zero.rank(), 0);
    let device_zero = rank_zero.to_cuda().unwrap();
    let (transposed, counters) = delta(|| device_zero.transpose().unwrap());
    assert_eq!(counters, CudaTransferStats::default(), "{counters:?}");
    assert_eq!(transposed.to_host().unwrap().data(), rank_zero.data());
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_transform_rejections_happen_before_any_device_work() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let device = fixture(&runtime).to_cuda().unwrap();
    // Warm the structures first, so a rejection cannot be confused with the
    // cold-path uploads of a neighbouring call.
    let _ = device.permute(&[2, 0], &[1, 3]).unwrap();
    let before = runtime.cuda_tree_transform_stats().unwrap();

    let host = fixture(&runtime);
    // Each rejection is written once and expanded against both receivers, so
    // the device error *mapping* is gated, not merely "an error came back".
    macro_rules! rejects_like_host {
        (|$t:ident| $call:expr) => {{
            let expected = {
                let $t = &host;
                $call
            }
            .unwrap_err()
            .to_string();
            let actual = {
                let $t = &device;
                $call
            }
            .unwrap_err()
            .to_string();
            assert_eq!(actual, expected, "device error text must be the Host's");
            actual
        }};
    }

    let (errors, counters) = delta(|| {
        [
            // The braid levels-length check precedes the identity detection,
            // exactly as on Host: identity axes with a short levels list is an
            // error, not a clone.
            rejects_like_host!(|t| t.braid(&[0, 1], &[2, 3], &[1, 2, 3])),
            rejects_like_host!(|t| t.braid(&[2, 0], &[1, 3], &[1, 2, 3, 4, 5])),
            // Malformed axes come back from the expert layer.
            rejects_like_host!(|t| t.permute(&[0, 0], &[2, 3])),
            // A non-planar re-arrangement is refused rather than braided.
            rejects_like_host!(|t| t.transpose_axes(&[1, 2], &[3, 0])),
            // A split beyond the rank has no planar reading.
            rejects_like_host!(|t| t.repartition(5)),
        ]
    });
    assert!(
        errors[0].contains("one level per source axis"),
        "{}",
        errors[0]
    );
    assert!(errors[0].contains("expected 4"), "{}", errors[0]);
    assert!(errors[4].contains("exceeds rank 4"), "{}", errors[4]);
    assert_eq!(
        counters,
        CudaTransferStats::default(),
        "a rejected transform must submit nothing: {counters:?}"
    );
    assert_eq!(runtime.cuda_tree_transform_stats().unwrap(), before);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_cold_device_transform_uploads_one_coefficient_payload_per_structure() {
    // SU(2) recouples, so the structure has a real coefficient payload and a
    // pack/scatter workspace. Both are uploaded once, on the first replay of
    // that structure, and never again while the Host store admits it.
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let v = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::SU2FusionRule),
        [
            (tenet::core::SU2Irrep::from_twice_spin(0), 2),
            (tenet::core::SU2Irrep::from_twice_spin(1), 2),
            (tenet::core::SU2Irrep::from_twice_spin(2), 1),
        ],
    )
    .unwrap();
    let host: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&v, &v], [&v, &v], |_, indices| {
            indices.iter().map(|&i| i as f64 + 1.0).sum::<f64>()
        })
        .unwrap();
    let output_bytes = std::mem::size_of_val(host.permute(&[1, 2], &[3, 0]).unwrap().data()) as u64;
    let device = host.to_cuda().unwrap();

    let before = runtime.cuda_tree_transform_stats().unwrap();
    let (_, cold) = delta(|| device.permute(&[1, 2], &[3, 0]).unwrap());
    let after = runtime.cuda_tree_transform_stats().unwrap();
    assert!(
        cold.h2d_calls >= 2,
        "a cold recoupling replay uploads the output and its coefficient payload: {cold:?}"
    );
    assert_eq!(cold.d2h_calls, 0, "{cold:?}");
    assert_eq!(after.prepared_structures, before.prepared_structures + 1);
    assert!(
        after.executor_bytes > before.executor_bytes,
        "the coefficient payload and workspace are retained: {before:?} -> {after:?}"
    );
    assert!(after.workspace_bytes > 0, "{after:?}");

    // Warm: the #740 output initialisation and nothing else. No second payload,
    // no workspace growth.
    let (_, warm) = delta(|| device.permute(&[1, 2], &[3, 0]).unwrap());
    assert_eq!(warm.h2d_calls, 1, "{warm:?}");
    assert_eq!(warm.h2d_bytes, output_bytes, "{warm:?}");
    assert_eq!(warm.d2h_calls, 0, "{warm:?}");
    assert_eq!(warm.device_allocs, 1, "{warm:?}");
    assert_eq!(runtime.cuda_tree_transform_stats().unwrap(), after);
}
