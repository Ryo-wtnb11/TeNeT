//! Device replay of *provider-compiled* recoupling structures (issue #1310).
//!
//! The structure-level device suite in `tenet-operations` proves the executor's
//! algebra on hand-written fixtures. This one proves it on the structures the
//! categorical layer actually compiles — SU(2) and fZ2 x SU(2) F moves reached
//! through `TreeTransformCache`, one level below the typed `TensorMap` API —
//! whose recoupling matrices carry irrational 6j entries, mixed signs, and, for
//! the product rule, fermionic signs.
//!
//! Every value is compared with the structure-independent oracle in
//! `categorical_recoupling` *and* with the host executor replaying the same
//! compiled structure. The device is the unit under test, so every test is
//! `#[ignore]` like the rest of the device suite.

#![cfg(feature = "cuda")]

mod categorical_recoupling;

use categorical_recoupling::{
    assert_close, expected, fixtures, host_replay, non_symmetric_fixture, Compiled, TestScalar,
};
use num_complex::Complex64;
use tenet_dense::{cuda_transfer_stats, CudaDenseContext, CudaScalar, CudaTransferStats};
use tenet_operations::cuda::CudaStorage;
use tenet_operations::{CudaTreeTransformDestination, CudaTreeTransformExecutor};

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

fn device_replay<T: DeviceScalar>(
    ctx: &mut CudaDenseContext,
    executor: &mut CudaTreeTransformExecutor,
    fixture: &Compiled,
    source: &[T],
    destination: &[T],
    overwrite: bool,
) -> Vec<T> {
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
            &fixture.structure,
            &fixture.space,
            &fixture.space,
            &mut device_dst,
            &device_src,
            mode,
        )
        .unwrap();
    device_dst.download(ctx).unwrap()
}

fn check<T: DeviceScalar>(
    ctx: &mut CudaDenseContext,
    executor: &mut CudaTreeTransformExecutor,
    fixture: &Compiled,
) {
    let source = fixture.source::<T>();
    let destination: Vec<T> = (0..fixture.len())
        .map(|index| T::from_parts(-3.0 - index as f64, 0.5))
        .collect();
    for overwrite in [true, false] {
        let what = format!("{} / {} / overwrite = {overwrite}", fixture.name, T::NAME);
        let device = device_replay(ctx, executor, fixture, &source, &destination, overwrite);
        assert_close(
            &device,
            &expected(fixture, &source, &destination, overwrite),
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
fn device_replay_matches_provider_compiled_recoupling_structures() {
    // What: SU(2) and fZ2 x SU(2) permutes, braids and planar transposes at
    // ranks 4 and 6, with two- and five-channel recoupling, degeneracy 1 and 2,
    // conjugated sources, both payload dtypes and both destination modes.
    let mut ctx = context();
    let mut executor = CudaTreeTransformExecutor::default();
    for fixture in fixtures() {
        check::<f64>(&mut ctx, &mut executor, &fixture);
        check::<Complex64>(&mut ctx, &mut executor, &fixture);
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn the_device_applies_the_transposed_recoupling_matrix_on_provider_data() {
    // Negative control for the GEMM orientation, on the device and on real 6j
    // data: the fixture's five-channel matrix is not its own transpose, so a
    // device that contracted `U` instead of `Uᵀ` would produce the other
    // buffer. The CPU-only suite proves the two buffers differ; this asserts
    // the device lands on the oracle's one.
    let mut ctx = context();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = non_symmetric_fixture();
    assert!(
        fixture.has_non_symmetric_matrix(),
        "the orientation control needs a non-symmetric U"
    );
    let source = fixture.source::<f64>();
    let destination = vec![0.0_f64; fixture.len()];

    let device = device_replay(
        &mut ctx,
        &mut executor,
        &fixture,
        &source,
        &destination,
        true,
    );

    assert_close(
        &device,
        &expected(&fixture, &source, &destination, true),
        "su2_rank6_permute: device vs oracle",
    );
    assert_close(
        &device,
        &host_replay(&fixture, &source, &destination, true),
        "su2_rank6_permute: device vs host",
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
fn a_warm_provider_structure_replay_transfers_nothing() {
    // What: the zero-transfer warm-replay contract holds for a structure the
    // categorical layer compiled, not only for hand-written fixtures — the
    // uploaded coefficient payload and the pack/scatter workspace are the whole
    // per-structure device state.
    let mut ctx = context();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = non_symmetric_fixture();
    let source = fixture.source::<f64>();
    let destination = vec![0.0_f64; fixture.len()];
    let mut device_dst = CudaStorage::<f64>::upload(&ctx, &destination).unwrap();
    let device_src = CudaStorage::<f64>::upload(&ctx, &source).unwrap();
    let replay = |ctx: &mut CudaDenseContext,
                  executor: &mut CudaTreeTransformExecutor,
                  dst: &mut CudaStorage<f64>| {
        executor
            .replay(
                ctx,
                &fixture.structure,
                &fixture.space,
                &fixture.space,
                dst,
                &device_src,
                CudaTreeTransformDestination::Overwrite,
            )
            .unwrap();
    };

    replay(&mut ctx, &mut executor, &mut device_dst);
    let cold = cuda_transfer_stats();
    let cold_workspace = executor.workspace_device_bytes();
    assert!(
        cold_workspace > 0,
        "a recoupling structure needs a workspace"
    );
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
    assert!(warm.gemm_calls > 0, "warm replay submitted nothing");
    assert_close(
        &device_dst.download(&ctx).unwrap(),
        &expected(&fixture, &source, &destination, true),
        "warm provider-structure replay",
    );
}
