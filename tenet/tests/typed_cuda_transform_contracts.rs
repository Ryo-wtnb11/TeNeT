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
use tenet::prelude::Error;
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

// ---------------------------------------------------------------------------
// `*_overwrite_into` contracts (issue #1329, G2b-3)
// ---------------------------------------------------------------------------

/// The device wording of a Host message. The device mirrors Host's error
/// variants, messages and order; only the storage noun names the placement,
/// as in `contract_overwrite_into_with_template`. That these are the exact
/// Host strings is pinned without a device in `typed_transform_host_side.rs`.
fn as_device_message(host: &str) -> String {
    host.replace("ordinary dense host source", "ordinary dense CUDA source")
        .replace("ordinary dense host storage", "ordinary dense CUDA storage")
}

/// The warm contract: no transfer in either direction and no device
/// allocation. Kernel submission counts (`gemm_calls`, `copy_calls`) are not
/// transfers and are expected to be non-zero.
fn assert_transfer_free(stats: &CudaTransferStats) {
    assert_eq!(stats.h2d_calls, 0, "{stats:?}");
    assert_eq!(stats.h2d_bytes, 0, "{stats:?}");
    assert_eq!(stats.d2h_calls, 0, "{stats:?}");
    assert_eq!(stats.d2h_bytes, 0, "{stats:?}");
    assert_eq!(stats.device_allocs, 0, "{stats:?}");
}

/// A device payload as raw bits, so a NaN-poisoned destination compares equal
/// to itself.
fn payload_bits<R>(tensor: &TensorMap<R, f64, tenet::typed::CudaStorage>) -> Vec<u64>
where
    R: tenet::core::MultiplicityFreeRigidSymbols<Scalar = f64>
        + tenet::core::CheckedFusionAlgebra
        + tenet::core::SectorCodec,
{
    tensor
        .to_host()
        .unwrap()
        .data()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_warm_device_overwrite_into_transfers_nothing_and_allocates_nothing() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let source = fixture(&runtime).to_cuda().unwrap();
    let mut destination = fixture(&runtime)
        .permute(&[2, 0], &[1, 3])
        .unwrap()
        .to_cuda()
        .unwrap();

    // Cold: prepares the structure for this pair.
    source
        .permute_overwrite_into(&mut destination, &[2, 0], &[1, 3], 1.0)
        .unwrap();
    let cold = runtime.cuda_tree_transform_stats().unwrap();

    // Warm: the destination is the caller's, so unlike the returning
    // `permute` there is no #740 output initialisation left to pay.
    let (_, warm) = delta(|| {
        source
            .permute_overwrite_into(&mut destination, &[2, 0], &[1, 3], -2.5)
            .unwrap()
    });
    assert_eq!(warm.h2d_calls, 0, "{warm:?}");
    assert_eq!(warm.h2d_bytes, 0, "{warm:?}");
    assert_eq!(warm.d2h_calls, 0, "{warm:?}");
    assert_eq!(warm.d2h_bytes, 0, "{warm:?}");
    assert_eq!(warm.device_allocs, 0, "{warm:?}");
    assert_eq!(
        runtime.cuda_tree_transform_stats().unwrap(),
        cold,
        "a warm overwrite grows no device state"
    );

    // A zero caller scale takes the zero-operand route. Its 1x1 operand is
    // element 0 of the context's zero template, so the *first* zero-scale
    // replay on a context whose template is shorter than one element sizes it
    // — one 8-byte upload and one device allocation, once per context, not per
    // call. Warm, it transfers nothing like any other scale.
    let (_, first_zero) = delta(|| {
        source
            .permute_overwrite_into(&mut destination, &[2, 0], &[1, 3], 0.0)
            .unwrap()
    });
    assert!(
        first_zero.h2d_calls <= 1
            && first_zero.h2d_bytes <= std::mem::size_of::<f64>() as u64
            && first_zero.device_allocs <= 1,
        "the zero template is one element uploaded once, not a buffer: {first_zero:?}"
    );
    let (_, zero) = delta(|| {
        source
            .permute_overwrite_into(&mut destination, &[2, 0], &[1, 3], -0.0)
            .unwrap()
    });
    assert_eq!(zero.h2d_calls, 0, "{zero:?}");
    assert_eq!(zero.d2h_calls, 0, "{zero:?}");
    assert_eq!(zero.device_allocs, 0, "{zero:?}");

    // And the other three methods are warm on their own structures.
    let mut transposed = fixture(&runtime).transpose().unwrap().to_cuda().unwrap();
    source
        .transpose_overwrite_into(&mut transposed, 1.0)
        .unwrap();
    let (_, warm) = delta(|| {
        source
            .transpose_overwrite_into(&mut transposed, 1.0)
            .unwrap()
    });
    assert_transfer_free(&warm);

    let mut bent = fixture(&runtime).repartition(1).unwrap().to_cuda().unwrap();
    source.repartition_overwrite_into(&mut bent, 1.0).unwrap();
    let (_, warm) = delta(|| source.repartition_overwrite_into(&mut bent, 1.0).unwrap());
    assert_transfer_free(&warm);

    let mut cyclic = fixture(&runtime)
        .transpose_axes(&[1, 3], &[0, 2])
        .unwrap()
        .to_cuda()
        .unwrap();
    source
        .transpose_axes_overwrite_into(&mut cyclic, &[1, 3], &[0, 2], 1.0)
        .unwrap();
    let (_, warm) = delta(|| {
        source
            .transpose_axes_overwrite_into(&mut cyclic, &[1, 3], &[0, 2], 1.0)
            .unwrap()
    });
    assert_transfer_free(&warm);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_overwrite_into_admits_the_exact_layout_on_the_shared_runtime_store() {
    // Host `overwrite_tree_transform` admits the source/destination layout
    // pair after a successful replay, so a second call resolves the operation
    // out of the store instead of deriving it. The store is the Runtime's own
    // Host store — the device path must feed and reuse exactly that one.
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let host = fixture(&runtime);
    let expected = host.permute(&[2, 0], &[1, 3]).unwrap();
    let source = host.to_cuda().unwrap();

    runtime.clear_tree_transform_cache();
    let mut first = expected.to_cuda().unwrap();
    source
        .permute_overwrite_into(&mut first, &[2, 0], &[1, 3], 1.0)
        .unwrap();
    let cold = runtime.tree_transform_cache_info();

    let mut second = expected.to_cuda().unwrap();
    source
        .permute_overwrite_into(&mut second, &[2, 0], &[1, 3], 1.0)
        .unwrap();
    let warm = runtime.tree_transform_cache_info();
    assert_eq!(
        warm.entries(),
        cold.entries(),
        "the admitted pair adds no entry"
    );
    assert!(
        warm.hits() > cold.hits(),
        "the second call must take the admitted path: {cold:?} -> {warm:?}"
    );
    assert_eq!(
        first.to_host().unwrap().data(),
        second.to_host().unwrap().data()
    );

    // Admission is keyed on the layout pair, so a destination that is *not*
    // this operation's result still misses the lookup and is still rejected.
    let mut wrong = host.transpose().unwrap().to_cuda().unwrap();
    let error = source
        .permute_overwrite_into(&mut wrong, &[2, 0], &[1, 3], 1.0)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("does not match the operation result"),
        "{error}"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_overwrite_into_rejections_happen_before_any_device_work() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let host = fixture(&runtime);
    let expected = host.permute(&[2, 0], &[1, 3]).unwrap();
    let source = host.to_cuda().unwrap();
    // Fresh, uniquely owned destinations on demand: a `clone()` would share
    // the payload `Arc` and trip the unique-ownership check instead.
    let host_destination = || host.permute(&[2, 0], &[1, 3]).unwrap();
    let device_destination = || expected.to_cuda().unwrap();

    // Warm the structure first, so a rejection cannot be confused with the
    // cold-path uploads of a neighbouring call.
    {
        let mut warm = device_destination();
        source
            .permute_overwrite_into(&mut warm, &[2, 0], &[1, 3], 1.0)
            .unwrap();
    }
    let stats_before = runtime.cuda_tree_transform_stats().unwrap();

    // Asserts one device rejection: the Host's error text (with the placement
    // noun), no transfer, allocation or executor-state change, and a
    // byte-identical witness — the owned device tensor whose payload the
    // destination is (itself, or, for a lazy destination, its parent).
    macro_rules! rejects_like_host {
        ($what:expr, $host_call:expr, $witness:ident, $device_call:expr) => {{
            let expected_message = as_device_message(&$host_call.unwrap_err().to_string());
            // Bitwise: the poisoned destinations are NaN, which is not equal
            // to itself.
            let before = payload_bits(&$witness);
            let (error, counters) = delta(|| $device_call.unwrap_err());
            assert_eq!(error.to_string(), expected_message, "{}", $what);
            assert_eq!(
                counters,
                CudaTransferStats::default(),
                "{}: a rejected overwrite must submit nothing: {counters:?}",
                $what
            );
            assert_eq!(
                payload_bits(&$witness),
                before,
                "{}: the caller's destination must be untouched",
                $what
            );
            error
        }};
    }

    // `permute_overwrite_into` takes `&mut Self`, so a rule mismatch is only
    // reachable between two *instances* of the same rule type — Z2 against
    // Z3, exactly as the Host unit test builds it.
    let z2 = Arc::new(tenet::core::ZNFusionRule::new(2).unwrap());
    let z3 = Arc::new(tenet::core::ZNFusionRule::new(3).unwrap());
    let z2_leg = GradedSpace::try_new_with_arc(Arc::clone(&z2), [(z2.irrep(0), 2)]).unwrap();
    let z3_leg = GradedSpace::try_new_with_arc(Arc::clone(&z3), [(z3.irrep(0), 2)]).unwrap();
    let zn_fill = |_: &_, indices: &[usize]| indices.iter().map(|&i| i as f64 + 1.0).sum::<f64>();
    let z2_host = TensorMap::from_block_fn(&runtime, [&z2_leg], [&z2_leg], zn_fill).unwrap();
    let z2_source = z2_host.to_cuda().unwrap();

    // 1. Runtime mismatch wins over a rule mismatch: a Z3 destination on
    //    another Runtime.
    let other = Runtime::builder().cuda(0).build().unwrap();
    let mut host_foreign =
        TensorMap::from_block_fn(&other, [&z3_leg], [&z3_leg], |_, _| f64::NAN).unwrap();
    let mut foreign = host_foreign.to_cuda().unwrap();
    {
        let witness = foreign.clone();
        let error = rejects_like_host!(
            "runtime mismatch precedes rule mismatch",
            z2_host.permute_overwrite_into(&mut host_foreign, &[0], &[1], 1.0),
            witness,
            z2_source.permute_overwrite_into(&mut foreign, &[0], &[1], 1.0)
        );
        assert_eq!(error, Error::RuntimeMismatch);
    }

    // 2. Rule mismatch wins over a lazy-adjoint source.
    let mut host_z3 =
        TensorMap::from_block_fn(&runtime, [&z3_leg], [&z3_leg], |_, _| f64::NAN).unwrap();
    let mut z3_destination = host_z3.to_cuda().unwrap();
    let lazy_source = source.adjoint().unwrap();
    let host_lazy_source = host.adjoint().unwrap();
    let z2_lazy_source = z2_source.adjoint().unwrap();
    let z2_host_lazy_source = z2_host.adjoint().unwrap();
    {
        let witness = z3_destination.clone();
        let error = rejects_like_host!(
            "rule mismatch precedes the lazy-adjoint source",
            z2_host_lazy_source.permute_overwrite_into(&mut host_z3, &[0], &[1], 1.0),
            witness,
            z2_lazy_source.permute_overwrite_into(&mut z3_destination, &[0], &[1], 1.0)
        );
        assert_eq!(error, Error::RuleMismatch);
    }

    // 3. A lazy-adjoint source is rejected, never lowered onto its parent —
    //    and it is reported before a lazy-adjoint *destination*.
    let lazy_parent = device_destination();
    let mut lazy_destination = lazy_parent.adjoint().unwrap();
    let mut host_lazy_destination = host_destination().adjoint().unwrap();
    let error = rejects_like_host!(
        "the lazy-adjoint source precedes the lazy-adjoint destination",
        host_lazy_source.permute_overwrite_into(&mut host_lazy_destination, &[2, 0], &[1, 3], 1.0),
        lazy_parent,
        lazy_source.permute_overwrite_into(&mut lazy_destination, &[2, 0], &[1, 3], 1.0)
    );
    assert!(
        error.to_string().contains("ordinary dense CUDA source"),
        "{error}"
    );

    // 4. A lazy-adjoint destination alone.
    let mut host_lazy_destination = host_destination().adjoint().unwrap();
    let error = rejects_like_host!(
        "lazy-adjoint destination",
        host.permute_overwrite_into(&mut host_lazy_destination, &[2, 0], &[1, 3], 1.0),
        lazy_parent,
        source.permute_overwrite_into(&mut lazy_destination, &[2, 0], &[1, 3], 1.0)
    );
    assert!(
        error.to_string().contains("ordinary dense CUDA storage"),
        "{error}"
    );

    // 5. Alias through a clone: `TensorMap` clones share the payload `Arc`, so
    //    source and destination can be the same device allocation. The alias
    //    check precedes the operation build, so malformed axes do not mask it.
    let mut alias = source.clone();
    let mut host_alias = host.clone();
    {
        let witness = source.clone();
        let error = rejects_like_host!(
            "the alias check precedes the operation build",
            host.permute_overwrite_into(&mut host_alias, &[0, 0], &[1, 3], 1.0),
            witness,
            source.permute_overwrite_into(&mut alias, &[0, 0], &[1, 3], 1.0)
        );
        assert!(error.to_string().contains("must not alias"), "{error}");
    }

    // 6. A destination on the wrong space, held shared: the space check
    //    precedes the unique-ownership check.
    let mut wrong_space = host.transpose().unwrap().to_cuda().unwrap();
    let wrong_space_handle = wrong_space.clone();
    let mut host_wrong_space = host.transpose().unwrap();
    let host_wrong_space_handle = host_wrong_space.clone();
    {
        let witness = wrong_space.clone();
        let error = rejects_like_host!(
            "the space check precedes unique ownership",
            host.permute_overwrite_into(&mut host_wrong_space, &[2, 0], &[1, 3], 1.0),
            witness,
            source.permute_overwrite_into(&mut wrong_space, &[2, 0], &[1, 3], 1.0)
        );
        assert!(
            error
                .to_string()
                .contains("does not match the operation result"),
            "{error}"
        );
    }
    drop(wrong_space_handle);
    drop(host_wrong_space_handle);

    // 7. A shared destination on the right space.
    let mut shared = device_destination();
    let shared_handle = shared.clone();
    let mut host_shared = host_destination();
    let host_shared_handle = host_shared.clone();
    {
        let witness = shared.clone();
        let error = rejects_like_host!(
            "shared destination",
            host.permute_overwrite_into(&mut host_shared, &[2, 0], &[1, 3], 1.0),
            witness,
            source.permute_overwrite_into(&mut shared, &[2, 0], &[1, 3], 1.0)
        );
        assert!(error.to_string().contains("uniquely owned"), "{error}");
    }
    drop(shared_handle);
    drop(host_shared_handle);

    // 8. Malformed axes, a non-planar cyclic request and a `repartition`
    //    destination of another rank all come back from the operation build,
    //    before the lease.
    let mut destination = device_destination();
    let mut host_dst = host_destination();
    let witness = destination.clone();
    rejects_like_host!(
        "malformed axes",
        host.permute_overwrite_into(&mut host_dst, &[0, 0], &[1, 3], 1.0),
        witness,
        source.permute_overwrite_into(&mut destination, &[0, 0], &[1, 3], 1.0)
    );
    rejects_like_host!(
        "non-planar transpose_axes",
        host.transpose_axes_overwrite_into(&mut host_dst, &[1, 2], &[3, 0], 1.0),
        witness,
        source.transpose_axes_overwrite_into(&mut destination, &[1, 2], &[3, 0], 1.0)
    );

    let v = leg();
    let mut host_rank_three =
        TensorMap::from_block_fn(&runtime, [&v, &v], [&v], |_, _| f64::NAN).unwrap();
    let mut rank_three = host_rank_three.to_cuda().unwrap();
    {
        let witness = rank_three.clone();
        let error = rejects_like_host!(
            "repartition destination of another rank",
            host.repartition_overwrite_into(&mut host_rank_three, 1.0),
            witness,
            source.repartition_overwrite_into(&mut rank_three, 1.0)
        );
        assert!(
            error.to_string().contains("does not match source rank"),
            "{error}"
        );
    }

    assert_eq!(
        runtime.cuda_tree_transform_stats().unwrap(),
        stats_before,
        "no rejection may change the device executor state"
    );
}
