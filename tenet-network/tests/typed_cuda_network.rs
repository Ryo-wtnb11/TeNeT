#![cfg(feature = "cuda")]

use std::sync::Arc;

use tenet::core::{
    product_sector, CheckedFusionAlgebra, FermionParityFusionRule, FusionAlgebraError,
    MultiplicityFreeAdmissionMode, MultiplicityFreeRigidSymbols, ProductFusionRuleExt,
    SU2FusionRule, SU2Irrep, SectorCodec, TypedSectorAdmission, U1FusionRule, U1Irrep, Z2Irrep,
};
use tenet::prelude::Complex64;
use tenet::typed::{CudaStorage, GradedSpace, Runtime, TensorMap};
use tenet_network::{
    clear_plan_cache, configure_plan_cache, plan_cache_stats, tensor, ContractionPlan,
    ContractionStep, GreedyDenseOptimizer, Network, PlanCacheConfig, TemporaryLabel, TensorId,
};

fn labels(names: &[&str]) -> Vec<TemporaryLabel> {
    names.iter().copied().map(TemporaryLabel::from).collect()
}

fn pair_network() -> Network {
    Network::new(
        vec![labels(&["a", "k"]), labels(&["k", "b"])],
        vec![false, false],
        vec![Some(1), Some(1)],
        labels(&["a", "b"]),
        Some(1),
    )
    .unwrap()
}

fn assert_unsupported_cuda_network(error: tenet::prelude::Error) {
    match error {
        tenet::prelude::Error::Operation(error) => assert!(matches!(
            error.as_ref(),
            tenet::operations::OperationError::UnsupportedTensorContractScope { .. }
        )),
        other => panic!("expected unsupported tensor-contract scope, got {other:?}"),
    }
}

fn cuda_pair<R>(runtime: &Runtime, space: &GradedSpace<R>, seed: u64)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync,
{
    let lhs = TensorMap::<R, f64>::rand_with_seed(runtime, [space], [space], seed).unwrap();
    let rhs = TensorMap::<R, f64>::rand_with_seed(runtime, [space], [space], seed + 1).unwrap();
    let host_plan = pair_network()
        .plan(&[&lhs, &rhs], &GreedyDenseOptimizer)
        .unwrap();
    let lhs_cuda = lhs.to_cuda().unwrap();
    let rhs_cuda = rhs.to_cuda().unwrap();
    let cuda_refs: [&TensorMap<R, f64, CudaStorage>; 2] = [&lhs_cuda, &rhs_cuda];
    let cuda_plan = pair_network()
        .plan(&cuda_refs, &GreedyDenseOptimizer)
        .unwrap();
    assert_eq!(host_plan.plan().steps(), cuda_plan.plan().steps());
    let actual = cuda_plan.execute_cuda(&cuda_refs).unwrap();
    let before_macro = plan_cache_stats(runtime);
    let macro_actual = tensor!([a; b] = lhs_cuda[a; k] * rhs_cuda[k; b]).unwrap();
    let macro_warm = tensor!([a; b] = lhs_cuda[a; k] * rhs_cuda[k; b]).unwrap();
    let after_macro = plan_cache_stats(runtime);
    let manual = lhs_cuda.contract(&rhs_cuda, &[1], &[0], &[0, 1]).unwrap();
    let host_oracle = lhs.contract(&rhs, &[1], &[0], &[0, 1]).unwrap();
    assert_eq!(actual.placement(), lhs_cuda.placement());
    assert!(std::ptr::eq(actual.provider(), lhs_cuda.provider()));
    assert_eq!(actual.codomain(), host_oracle.codomain());
    assert_eq!(actual.domain(), host_oracle.domain());
    let actual_host = actual.to_host().unwrap();
    assert_eq!(actual_host.data(), manual.to_host().unwrap().data());
    assert_eq!(actual_host.data(), host_oracle.data());
    assert_eq!(macro_actual.to_host().unwrap().data(), host_oracle.data());
    assert_eq!(macro_warm.to_host().unwrap().data(), host_oracle.data());
    assert_eq!(macro_actual.placement(), lhs_cuda.placement());
    // The warm device replay reproduces the returning result exactly: identical
    // submissions to the same device kernels, so f64 is bitwise equal.
    assert_eq!(
        macro_warm.to_host().unwrap().data(),
        actual.to_host().unwrap().data()
    );
    // This provider contributes exactly one device pool for `(R, f64,
    // CudaStorage<f64>)`; the second call leases that pool's idle workspace.
    assert_eq!(
        after_macro.workspaces_created - before_macro.workspaces_created,
        1,
        "one device workspace per (provider, dtype, storage)"
    );
    assert!(
        after_macro.workspace_reuses > before_macro.workspace_reuses,
        "the warm replay must reuse the leased workspace, not build a new one"
    );
    // Each cached plan keeps at most two typed pools of at most two idle
    // workspaces, so the runtime-wide idle count stays inside that bound.
    assert!(
        after_macro.idle_workspaces >= 1 && after_macro.idle_workspaces <= 4 * after_macro.entries,
        "idle workspaces {} outside the plan-wide bound for {} entries",
        after_macro.idle_workspaces,
        after_macro.entries
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn cuda_macro_rejects_trace_before_cache_or_execution() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let space =
        GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let tensor = TensorMap::<_, f64>::rand_with_seed(&runtime, [&space], [&space], 748_090)
        .unwrap()
        .to_cuda()
        .unwrap();
    let error = tensor!([] = tensor[i; i]).unwrap_err();
    assert!(matches!(
        error,
        tenet::prelude::Error::UnsupportedOnDevice(_)
    ));
    assert_eq!(plan_cache_stats(&runtime).entries, 0);
    assert_eq!(plan_cache_stats(&runtime).workspaces_created, 0);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn noncanonical_cuda_macro_never_publishes_or_touches_cache_state() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let space =
        GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&space], [&space], 748_091).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&space], [&space], 748_092).unwrap();
    let a_cuda = a.to_cuda().unwrap();
    let b_cuda = b.to_cuda().unwrap();

    assert!(runtime.with_extension_slot(|slot| slot.is_none()));
    for _ in 0..2 {
        let error = tensor!([k; i] = a_cuda[i; j] * b_cuda[j; k]).unwrap_err();
        assert_unsupported_cuda_network(error);
        assert!(runtime.with_extension_slot(|slot| slot.is_none()));
    }

    // Publish the same structural plan from Host. CUDA must validate the
    // cached plan before alias/LRU promotion or hit-counter mutation.
    drop(tensor!([k; i] = a[i; j] * b[j; k]).unwrap());
    let preseeded = plan_cache_stats(&runtime);
    for _ in 0..2 {
        let error = tensor!([k; i] = a_cuda[i; j] * b_cuda[j; k]).unwrap_err();
        assert_unsupported_cuda_network(error);
        assert_eq!(plan_cache_stats(&runtime), preseeded);
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn canonical_cuda_network_provider_matrix_chain_and_lazy_conj() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let u1 = GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    cuda_pair(&runtime, &u1, 748_100);
    let su2 =
        GradedSpace::try_new_with_arc(Arc::new(SU2FusionRule), [(SU2Irrep::from_twice_spin(0), 2)])
            .unwrap();
    cuda_pair(&runtime, &su2, 748_110);
    let fz2 = GradedSpace::try_new_with_arc(Arc::new(FermionParityFusionRule), [(Z2Irrep::ODD, 2)])
        .unwrap();
    cuda_pair(&runtime, &fz2, 748_120);
    let product = GradedSpace::try_new_with_arc(
        Arc::new(FermionParityFusionRule.product(U1FusionRule)),
        [(product_sector(Z2Irrep::ODD, U1Irrep::new(0)), 2)],
    )
    .unwrap();
    cuda_pair(&runtime, &product, 748_130);

    let tensors = (0..3)
        .map(|index| {
            TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&u1], [&u1], 748_140 + index)
                .unwrap()
                .to_cuda()
                .unwrap()
        })
        .collect::<Vec<_>>();
    let chain = Network::new(
        vec![
            labels(&["a", "b"]),
            labels(&["b", "c"]),
            labels(&["c", "d"]),
        ],
        vec![false; 3],
        vec![Some(1); 3],
        labels(&["a", "d"]),
        Some(1),
    )
    .unwrap();
    let refs = [&tensors[0], &tensors[1], &tensors[2]];
    let chain_order = ContractionPlan::new(
        3,
        labels(&["a", "d"]),
        vec![
            ContractionStep::new(
                TensorId::new(0),
                TensorId::new(1),
                TensorId::new(3),
                0,
                labels(&["a", "c"]),
            ),
            ContractionStep::new(
                TensorId::new(3),
                TensorId::new(2),
                TensorId::new(4),
                0,
                labels(&["a", "d"]),
            ),
        ],
    )
    .unwrap();
    let planned = chain.plan_with(&refs, chain_order).unwrap();
    let actual = planned.execute_cuda(&refs).unwrap();
    let manual = tensors[0]
        .contract(&tensors[1], &[1], &[0], &[0, 1])
        .unwrap()
        .contract(&tensors[2], &[1], &[0], &[0, 1])
        .unwrap();
    let chain_macro =
        tensor!([a; d] = (tensors[0])[a; b] * (tensors[1])[b; c] * (tensors[2])[c; d]).unwrap();
    assert_eq!(actual.codomain(), manual.codomain());
    assert_eq!(actual.domain(), manual.domain());
    assert_eq!(
        actual.to_host().unwrap().data(),
        manual.to_host().unwrap().data()
    );
    assert_eq!(
        chain_macro.to_host().unwrap().data(),
        manual.to_host().unwrap().data()
    );
    assert_eq!(chain_macro.placement(), tensors[0].placement());

    let conj_network = Network::new(
        vec![labels(&["k", "i"]), labels(&["k", "j"])],
        vec![true, false],
        vec![Some(1), Some(1)],
        labels(&["i", "j"]),
        Some(1),
    )
    .unwrap();
    let conj_refs = [&tensors[0], &tensors[1]];
    let conj_actual = conj_network
        .plan(&conj_refs, &GreedyDenseOptimizer)
        .unwrap()
        .execute_cuda(&conj_refs)
        .unwrap();
    let conj_manual = tensors[0]
        .adjoint()
        .unwrap()
        .contract(&tensors[1], &[1], &[0], &[0, 1])
        .unwrap();
    let conj_macro = tensor!([i; j] = conj((tensors[0]))[k; i] * (tensors[1])[k; j]).unwrap();
    assert_eq!(conj_actual.codomain(), conj_manual.codomain());
    assert_eq!(conj_actual.domain(), conj_manual.domain());
    assert_eq!(
        conj_actual.to_host().unwrap().data(),
        conj_manual.to_host().unwrap().data()
    );
    assert_eq!(
        conj_macro.to_host().unwrap().data(),
        conj_manual.to_host().unwrap().data()
    );

    let single = Network::new(
        vec![labels(&["k", "i"])],
        vec![true],
        vec![Some(1)],
        labels(&["i", "k"]),
        Some(1),
    )
    .unwrap();
    let single_actual = single
        .plan(&[&tensors[0]], &GreedyDenseOptimizer)
        .unwrap()
        .execute_cuda(&[&tensors[0]])
        .unwrap();
    let single_expected = tensors[0].adjoint().unwrap();
    let single_macro = tensor!([i; k] = conj(tensors[0])[k; i]).unwrap();
    assert!(std::ptr::eq(
        single_actual.provider(),
        tensors[0].provider()
    ));
    assert_eq!(single_actual.codomain(), single_expected.codomain());
    assert_eq!(single_actual.domain(), single_expected.domain());
    assert_eq!(
        single_actual.to_host().unwrap().data(),
        single_expected.to_host().unwrap().data()
    );
    assert_eq!(
        single_macro.to_host().unwrap().data(),
        single_expected.to_host().unwrap().data()
    );

    let ket = TensorMap::<U1FusionRule, f64>::rand_with_seed(
        &runtime,
        std::iter::empty::<&GradedSpace<U1FusionRule>>(),
        [&u1],
        748_150,
    )
    .unwrap()
    .to_cuda()
    .unwrap();
    let bra = TensorMap::<U1FusionRule, f64>::rand_with_seed(
        &runtime,
        [&u1],
        std::iter::empty::<&GradedSpace<U1FusionRule>>(),
        748_151,
    )
    .unwrap()
    .to_cuda()
    .unwrap();
    let scalar_network = Network::new(
        vec![labels(&["k"]), labels(&["k"])],
        vec![false; 2],
        vec![Some(0), Some(1)],
        vec![],
        Some(0),
    )
    .unwrap();
    let scalar = scalar_network
        .plan(&[&ket, &bra], &GreedyDenseOptimizer)
        .unwrap()
        .execute_cuda(&[&ket, &bra])
        .unwrap();
    let scalar_macro = tensor!([] = ket[; k] * bra[k;]).unwrap();
    assert_eq!(scalar.rank(), 0);
    assert!(scalar.codomain().is_empty());
    assert!(scalar.domain().is_empty());
    assert!(std::ptr::eq(scalar.provider(), ket.provider()));
    assert_eq!(
        scalar_macro.to_host().unwrap().data(),
        scalar.to_host().unwrap().data()
    );
    let stats = plan_cache_stats(&runtime);
    assert!(
        stats.workspaces_created >= 4,
        "each of the four providers leases its own device workspace"
    );
    assert!(
        stats.idle_workspaces <= 4 * stats.entries,
        "idle workspaces {} outside the plan-wide bound for {} entries",
        stats.idle_workspaces,
        stats.entries
    );
}

/// G1a (#1268): canonical `tensor!` device execution with a genuinely complex
/// payload, including a lazy conjugate operand. Noncanonical routes and
/// intra-operand trace stay rejected without publishing a plan.
#[test]
#[ignore = "requires a real CUDA device"]
fn canonical_cuda_network_executes_complex_payloads_and_still_rejects_the_rest() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let u1 = GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let complex_entry = |indices: &[usize], seed: f64| {
        let ramp = indices.iter().map(|&index| index as f64).sum::<f64>();
        Complex64::new(ramp + seed, -(ramp + seed + 0.75))
    };
    let host: Vec<TensorMap<U1FusionRule, Complex64>> = (0..3)
        .map(|index| {
            TensorMap::from_block_fn(&runtime, [&u1], [&u1], |_, indices| {
                complex_entry(indices, 1.0 + index as f64)
            })
            .unwrap()
        })
        .collect();
    let device: Vec<_> = host
        .iter()
        .map(|tensor| tensor.to_cuda().unwrap())
        .collect();

    let refs: [&TensorMap<U1FusionRule, Complex64, CudaStorage<Complex64>>; 2] =
        [&device[0], &device[1]];
    let planned = pair_network().plan(&refs, &GreedyDenseOptimizer).unwrap();
    let executed = planned.execute_cuda(&refs).unwrap();
    let host_oracle = host[0].contract(&host[1], &[1], &[0], &[0, 1]).unwrap();
    assert_eq!(executed.placement(), device[0].placement());
    assert_close_c64(executed.to_host().unwrap().data(), host_oracle.data());

    let chain =
        tensor!([a; d] = (device[0])[a; b] * (device[1])[b; c] * (device[2])[c; d]).unwrap();
    let chain_oracle = host[0]
        .contract(&host[1], &[1], &[0], &[0, 1])
        .unwrap()
        .contract(&host[2], &[1], &[0], &[0, 1])
        .unwrap();
    assert_close_c64(chain.to_host().unwrap().data(), chain_oracle.data());
    let chain_warm =
        tensor!([a; d] = (device[0])[a; b] * (device[1])[b; c] * (device[2])[c; d]).unwrap();
    assert_close_c64(chain_warm.to_host().unwrap().data(), chain_oracle.data());

    let conj = tensor!([i; j] = conj((device[0]))[k; i] * (device[1])[k; j]).unwrap();
    let conj_oracle = host[0]
        .adjoint()
        .unwrap()
        .contract(&host[1], &[1], &[0], &[0, 1])
        .unwrap();
    assert_close_c64(conj.to_host().unwrap().data(), conj_oracle.data());

    let trace_error = tensor!([] = (device[0])[i; i]).unwrap_err();
    assert!(matches!(
        trace_error,
        tenet::prelude::Error::UnsupportedOnDevice(_)
    ));
    assert_unsupported_cuda_network(
        tensor!([k; i] = (device[0])[i; j] * (device[1])[j; k]).unwrap_err(),
    );
}

fn assert_close_c64(actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).norm() <= 1e-12 * (1.0 + expected.norm()),
            "actual {actual:?}, expected {expected:?}"
        );
    }
}

fn u1_pair(
    runtime: &Runtime,
    space: &GradedSpace<U1FusionRule>,
    seed: u64,
) -> (
    TensorMap<U1FusionRule, f64>,
    TensorMap<U1FusionRule, f64, CudaStorage>,
) {
    let host =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(runtime, [space], [space], seed).unwrap();
    let device = host.to_cuda().unwrap();
    (host, device)
}

/// G3c-1 (#1274): Host and device execution of one topology and one `(R, D)`
/// key separate workspace pools. A shared key would hand the device lease a
/// `NetworkExecutionWorkspace<_, _, Vec<f64>>` and panic in the registry
/// downcast.
#[test]
#[ignore = "requires a real CUDA device"]
fn host_and_cuda_macros_of_one_topology_use_separate_workspace_pools() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let u1 = GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let (a, a_cuda) = u1_pair(&runtime, &u1, 748_400);
    let (b, b_cuda) = u1_pair(&runtime, &u1, 748_401);

    let host = tensor!([i; k] = a[i; j] * b[j; k]).unwrap();
    let device = tensor!([i; k] = a_cuda[i; j] * b_cuda[j; k]).unwrap();
    assert_eq!(device.to_host().unwrap().data(), host.data());

    let stats = plan_cache_stats(&runtime);
    assert_eq!(stats.entries, 1, "one structural topology is shared");
    assert_eq!(
        stats.workspaces_created, 2,
        "storage is part of the pool key: one Host pool and one device pool"
    );
}

/// G3c-1 (#1274): a failed device step quarantines its lease instead of
/// returning buffers whose contents the failure may have disturbed, and the
/// next valid call rebuilds from a fresh workspace.
#[test]
#[ignore = "requires a real CUDA device"]
fn a_failed_cuda_step_quarantines_the_lease_and_the_next_call_rebuilds() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let u1 = GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let wrong =
        GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), 3)]).unwrap();
    let (_, a_cuda) = u1_pair(&runtime, &u1, 748_410);
    let (_, b_cuda) = u1_pair(&runtime, &u1, 748_411);
    let (_, mismatched) = u1_pair(&runtime, &wrong, 748_412);

    let warm = tensor!([i; k] = a_cuda[i; j] * b_cuda[j; k]).unwrap();
    let after_warm = plan_cache_stats(&runtime);
    assert_eq!(
        after_warm.idle_workspaces, 1,
        "a successful device call recycles its lease"
    );

    assert!(tensor!([i; k] = a_cuda[i; j] * mismatched[j; k]).is_err());
    let after_failure = plan_cache_stats(&runtime);
    assert_eq!(
        after_failure.idle_workspaces, 0,
        "the failed lease is quarantined, not recycled"
    );

    let rebuilt = tensor!([i; k] = a_cuda[i; j] * b_cuda[j; k]).unwrap();
    assert_eq!(
        rebuilt.to_host().unwrap().data(),
        warm.to_host().unwrap().data()
    );
    assert_eq!(
        plan_cache_stats(&runtime).workspaces_created,
        after_failure.workspaces_created + 1,
        "the quarantined workspace is replaced, not resurrected"
    );
}

/// G3c-1 (#1274), tightened by G3c-2 (#1276): shape drift that keeps the
/// payload length identical but changes the block layout still discards the
/// cached replay state. The length check alone would accept it; the per-operand
/// sector snapshot must not. Now that destinations are reused, a retained
/// device buffer of the other charge would survive the drift, so the drifted
/// result — and the undrifted result computed after it — are direct evidence
/// that the stale payload was dropped.
#[test]
#[ignore = "requires a real CUDA device"]
fn equal_length_block_layout_drift_discards_the_device_replay_state() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    // Both spaces carry one sector of degeneracy 2, so `[v; v]` has a single
    // 2x2 coupled block and exactly the same `required_len` either way.
    let charge_zero =
        GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let charge_three =
        GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(3), 2)]).unwrap();
    let (a0, a0_cuda) = u1_pair(&runtime, &charge_zero, 748_420);
    let (b0, b0_cuda) = u1_pair(&runtime, &charge_zero, 748_421);
    let (a3, a3_cuda) = u1_pair(&runtime, &charge_three, 748_422);
    let (b3, b3_cuda) = u1_pair(&runtime, &charge_three, 748_423);
    assert_eq!(a0.data().len(), a3.data().len());

    let first = tensor!([i; k] = a0_cuda[i; j] * b0_cuda[j; k]).unwrap();
    assert_eq!(
        first.to_host().unwrap().data(),
        a0.contract(&b0, &[1], &[0], &[0, 1]).unwrap().data()
    );
    let after_first = plan_cache_stats(&runtime);

    let drifted = tensor!([i; k] = a3_cuda[i; j] * b3_cuda[j; k]).unwrap();
    assert_eq!(
        drifted.to_host().unwrap().data(),
        a3.contract(&b3, &[1], &[0], &[0, 1]).unwrap().data()
    );

    let after_drift = plan_cache_stats(&runtime);
    assert_eq!(
        after_drift.workspaces_created, after_first.workspaces_created,
        "drift discards the replay state inside the pooled workspace, not the workspace"
    );
    assert!(
        after_drift.workspace_reuses > after_first.workspace_reuses,
        "the drifted call still leases the pooled device workspace"
    );

    // Back to the original charges: the workspace rebuilt for the drifted
    // layout must not leak into this one either.
    let restored = tensor!([i; k] = a0_cuda[i; j] * b0_cuda[j; k]).unwrap();
    assert_eq!(
        restored.to_host().unwrap().data(),
        first.to_host().unwrap().data()
    );
}

type ChainPair<R> = (Vec<TensorMap<R, f64>>, Vec<TensorMap<R, f64, CudaStorage>>);

/// A three-tensor canonical chain, which is the smallest schedule with a
/// reusable destination: step 0 overwrites the retained intermediate, step 1
/// produces the returned final output.
fn cuda_chain_tensors<R>(runtime: &Runtime, space: &GradedSpace<R>, seed: u64) -> ChainPair<R>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync,
{
    let host: Vec<_> = (0..3)
        .map(|index| {
            TensorMap::<R, f64>::rand_with_seed(runtime, [space], [space], seed + index).unwrap()
        })
        .collect();
    let device = host.iter().map(|t| t.to_cuda().unwrap()).collect();
    (host, device)
}

fn cuda_chain_reuse_matches_returning<R>(runtime: &Runtime, space: &GradedSpace<R>, seed: u64)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync,
{
    let (host, device) = cuda_chain_tensors(runtime, space, seed);
    let returning = device[0]
        .contract(&device[1], &[1], &[0], &[0, 1])
        .unwrap()
        .contract(&device[2], &[1], &[0], &[0, 1])
        .unwrap();
    let host_oracle = host[0]
        .contract(&host[1], &[1], &[0], &[0, 1])
        .unwrap()
        .contract(&host[2], &[1], &[0], &[0, 1])
        .unwrap();
    let cold = tensor!([a; d] = (device[0])[a; b] * (device[1])[b; c] * (device[2])[c; d]).unwrap();
    let warm = tensor!([a; d] = (device[0])[a; b] * (device[1])[b; c] * (device[2])[c; d]).unwrap();
    assert_eq!(cold.to_host().unwrap().data(), host_oracle.data());
    // The warm replay submits the same kernels in the same order to the same
    // device, so f64 is bitwise equal to the returning result. This relies on
    // cuBLAS/cuTENSOR determinism for identical submissions, as the existing
    // device equality assertions already do.
    assert_eq!(
        warm.to_host().unwrap().data(),
        returning.to_host().unwrap().data()
    );
}

/// G3c-2 (#1276): warm `tensor!` reuses its retained device destinations and
/// still returns exactly the returning contraction, for every admitted
/// categorical structure.
#[test]
#[ignore = "requires a real CUDA device"]
fn warm_cuda_destination_reuse_matches_the_returning_chain_for_every_provider() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let u1 = GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)],
    )
    .unwrap();
    cuda_chain_reuse_matches_returning(&runtime, &u1, 761_100);
    let su2 =
        GradedSpace::try_new_with_arc(Arc::new(SU2FusionRule), [(SU2Irrep::from_twice_spin(0), 2)])
            .unwrap();
    cuda_chain_reuse_matches_returning(&runtime, &su2, 761_110);
    let fz2 = GradedSpace::try_new_with_arc(Arc::new(FermionParityFusionRule), [(Z2Irrep::ODD, 2)])
        .unwrap();
    cuda_chain_reuse_matches_returning(&runtime, &fz2, 761_120);
    let product = GradedSpace::try_new_with_arc(
        Arc::new(FermionParityFusionRule.product(U1FusionRule)),
        [(product_sector(Z2Irrep::ODD, U1Irrep::new(0)), 2)],
    )
    .unwrap();
    cuda_chain_reuse_matches_returning(&runtime, &product, 761_130);

    // Complex payloads take the same route; the device kernels differ, so this
    // is a tolerance comparison against the Host oracle.
    let device: Vec<_> = (0..3)
        .map(|index| {
            TensorMap::<U1FusionRule, Complex64>::rand_with_seed(
                &runtime,
                [&u1],
                [&u1],
                761_140 + index,
            )
            .unwrap()
        })
        .collect();
    let cuda: Vec<_> = device.iter().map(|t| t.to_cuda().unwrap()).collect();
    let oracle = device[0]
        .contract(&device[1], &[1], &[0], &[0, 1])
        .unwrap()
        .contract(&device[2], &[1], &[0], &[0, 1])
        .unwrap();
    drop(tensor!([a; d] = (cuda[0])[a; b] * (cuda[1])[b; c] * (cuda[2])[c; d]).unwrap());
    let warm = tensor!([a; d] = (cuda[0])[a; b] * (cuda[1])[b; c] * (cuda[2])[c; d]).unwrap();
    assert_close_c64(warm.to_host().unwrap().data(), oracle.data());
}

/// G3c-2 (#1276): the warm device replay of an N-tensor chain performs no
/// host-to-device traffic and no device allocation for its N-2 intermediate
/// steps; each of them is reset by one D2D copy from the workspace's zero
/// template. Only the final, returned output still uploads its zeros
/// (#740/G3b).
///
/// The counters are process-wide, so this test must not run beside another
/// device test; the device suite runs with `--test-threads=1`.
#[test]
#[ignore = "requires a real CUDA device"]
fn warm_cuda_chain_uploads_nothing_for_its_reused_destinations() {
    use tenet::dense::cuda_transfer_stats;

    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let u1 = GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 4), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let (_, device) = cuda_chain_tensors(&runtime, &u1, 761_200);
    let (_, extra) = cuda_chain_tensors(&runtime, &u1, 761_300);

    // Four tensors: three steps, two of them over retained destinations.
    let run = || {
        tensor!(
            [a; e] = (device[0])[a; b]
                * (device[1])[b; c]
                * (device[2])[c; d]
                * (extra[0])[d; e]
        )
        .unwrap()
    };
    // Call 1 has nothing retained yet, so every step returns.
    drop(run());
    // Call 2 is the first to overwrite: it uploads the zero template once, on
    // top of the returned output's own zeros.
    let before_second = cuda_transfer_stats();
    drop(run());
    let after_second = cuda_transfer_stats();
    assert_eq!(
        (
            after_second.h2d_calls - before_second.h2d_calls,
            after_second.copy_calls - before_second.copy_calls,
        ),
        (2, 2),
        "the zero template costs exactly one upload, once per workspace"
    );

    // Call 3 onward is the steady state this contract describes.
    let before = cuda_transfer_stats();
    let warm = run();
    let after = cuda_transfer_stats();

    assert_eq!(
        (
            after.h2d_calls - before.h2d_calls,
            after.device_allocs - before.device_allocs,
            after.copy_calls - before.copy_calls,
            after.d2h_calls - before.d2h_calls,
            after.gemm_calls - before.gemm_calls,
        ),
        (1, 1, 2, 0, 6),
        "(h2d, device_allocs, d2d_copies, d2h, gemm) for the warm 4-tensor chain"
    );
    drop(warm);
}

/// G3c-2 (#1276): the device workspace's zero template is charged to the one
/// `workspace_budget_bytes` ledger, so a budget short by exactly the template
/// rejects the idle workspace whole.
#[test]
#[ignore = "requires a real CUDA device"]
fn the_device_zero_template_is_charged_to_the_workspace_budget() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    // One sector of degeneracy 8: every intermediate holds one 8x8 block, so
    // the template is exactly 64 f64 elements.
    let u1 = GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), 8)]).unwrap();
    let (_, device) = cuda_chain_tensors(&runtime, &u1, 761_400);
    let template_bytes = 8 * 8 * std::mem::size_of::<f64>();

    drop(tensor!([a; d] = (device[0])[a; b] * (device[1])[b; c] * (device[2])[c; d]).unwrap());
    let charge = plan_cache_stats(&runtime).retained_workspace_bytes;
    assert!(
        charge > template_bytes,
        "the device charge {charge} must include the {template_bytes}-byte template"
    );

    clear_plan_cache(&runtime);
    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            workspace_budget_bytes: charge - template_bytes,
            ..Default::default()
        },
    );
    drop(tensor!([a; d] = (device[0])[a; b] * (device[1])[b; c] * (device[2])[c; d]).unwrap());
    let rejected = plan_cache_stats(&runtime);
    assert_eq!(rejected.retained_workspace_bytes, 0);
    assert_eq!(rejected.idle_workspaces, 0);
    assert_eq!(rejected.workspace_byte_rejections, 1);

    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            workspace_budget_bytes: charge,
            ..Default::default()
        },
    );
    drop(tensor!([a; d] = (device[0])[a; b] * (device[1])[b; c] * (device[2])[c; d]).unwrap());
    let admitted = plan_cache_stats(&runtime);
    assert_eq!(admitted.retained_workspace_bytes, charge);
    assert_eq!(admitted.idle_workspaces, 1);
}
