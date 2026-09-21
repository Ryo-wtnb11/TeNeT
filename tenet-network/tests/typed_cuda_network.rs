#![cfg(feature = "cuda")]

use std::sync::Arc;

use tenet::core::{
    product_sector, CheckedFusionAlgebra, FermionParityFusionRule, FusionAlgebraError,
    MultiplicityFreeAdmissionMode, MultiplicityFreeRigidSymbols, ProductFusionRuleExt,
    SU2FusionRule, SU2Irrep, SectorCodec, TypedSectorAdmission, U1FusionRule, U1Irrep, Z2Irrep,
};
use tenet::prelude::{Complex32, Complex64};
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

/// G2c-5 (#1350): a full trace, the network with no contraction step, runs on
/// the device and equals the Host.
#[test]
#[ignore = "requires a real CUDA device"]
fn cuda_macro_full_trace_equals_host() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let space =
        GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let host = TensorMap::<_, f64>::rand_with_seed(&runtime, [&space], [&space], 748_090).unwrap();
    let tensor = host.to_cuda().unwrap();
    let device = tensor!([] = tensor[i; i]).unwrap();
    assert_eq!(device.placement(), tensor.placement());
    assert_close_dyn(
        device.to_host().unwrap().data(),
        tensor!([] = host[i; i]).unwrap().data(),
        f64::EPSILON,
        "full trace",
    );
}

/// G2c-3 (#1348): the reversed-output product the canonical predicate refused
/// runs on the device, equals the Host run, and shares the structural plan the
/// Host call published (one entry, a hit).
#[test]
#[ignore = "requires a real CUDA device"]
fn a_noncanonical_cuda_macro_runs_and_shares_the_host_plan() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let space =
        GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&space], [&space], 748_091).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&space], [&space], 748_092).unwrap();
    let a_cuda = a.to_cuda().unwrap();
    let b_cuda = b.to_cuda().unwrap();

    let host = tensor!([k; i] = a[i; j] * b[j; k]).unwrap();
    let published = plan_cache_stats(&runtime);
    assert_eq!(published.entries, 1);
    for _ in 0..2 {
        let device = tensor!([k; i] = a_cuda[i; j] * b_cuda[j; k]).unwrap();
        assert_eq!(device.codomain(), host.codomain());
        assert_eq!(device.domain(), host.domain());
        assert_close_dyn(
            device.to_host().unwrap().data(),
            host.data(),
            f64::EPSILON,
            "reversed pair",
        );
    }
    let after = plan_cache_stats(&runtime);
    assert_eq!(
        after.entries, 1,
        "Host and device share one structural plan"
    );
    assert!(after.hits > published.hits);
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
/// payload, including a lazy conjugate operand. Since G2c-3 (#1348) a
/// reversed output runs, and since G2c-5 (#1350) an intra-operand trace.
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

    let trace = tensor!([] = (device[0])[i; i]).unwrap();
    let trace_oracle = tensor!([] = (host[0])[i; i]).unwrap();
    assert_close_c64(trace.to_host().unwrap().data(), trace_oracle.data());
    // G2c-3 (#1348): the reversed output the canonical predicate refused runs.
    let reversed = tensor!([k; i] = (device[0])[i; j] * (device[1])[j; k]).unwrap();
    let reversed_oracle = tensor!([k; i] = (host[0])[i; j] * (host[1])[j; k]).unwrap();
    assert_close_c64(reversed.to_host().unwrap().data(), reversed_oracle.data());
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

/// G3c-2 (#1276), G2c-1b (#1346): the warm device replay of an N-tensor chain
/// performs no host-to-device traffic, no device allocation and no copy for
/// its N-2 intermediate steps: every destination block of this chain has a
/// GEMM, so a retained destination needs no reset at all. Only the final,
/// returned output still uploads its zeros (#740/G3b).
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
    // Call 2 is the first to overwrite: nothing but the returned output's own
    // zeros is uploaded, and nothing is reset.
    let before_second = cuda_transfer_stats();
    drop(run());
    let after_second = cuda_transfer_stats();
    assert_eq!(
        (
            after_second.h2d_calls - before_second.h2d_calls,
            after_second.copy_calls - before_second.copy_calls,
        ),
        (1, 0),
        "overwriting a retained destination uploads and copies nothing"
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
        (1, 1, 0, 0, 6),
        "(h2d, device_allocs, d2d_copies, d2h, gemm) for the warm 4-tensor chain"
    );
    drop(warm);
}

/// G3c-2 (#1276): an idle device workspace is charged to the one
/// `workspace_budget_bytes` ledger, so a budget one byte short of its charge
/// rejects it whole. Since G2c-1b (#1346) the charge holds no reset scratch:
/// a retained device destination is overwritten without a reset.
#[test]
#[ignore = "requires a real CUDA device"]
fn the_device_workspace_is_charged_to_the_workspace_budget() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let u1 = GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), 8)]).unwrap();
    let (_, device) = cuda_chain_tensors(&runtime, &u1, 761_400);

    drop(tensor!([a; d] = (device[0])[a; b] * (device[1])[b; c] * (device[2])[c; d]).unwrap());
    let charge = plan_cache_stats(&runtime).retained_workspace_bytes;
    assert!(
        charge > 0,
        "the idle device workspace retains its destinations"
    );

    clear_plan_cache(&runtime);
    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            workspace_budget_bytes: charge - 1,
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

// ---------------------------------------------------------------------------
// Single-precision device networks (leaf C2, #1336)
// ---------------------------------------------------------------------------

/// The device network path is generic over the payload dtype, so the chain
/// contracts have single-precision twins that run the same schedule. The
/// oracle is the host chain *at the same dtype*; the tolerance is the payload's
/// own, not `f64`'s.
type TypedChainPair<R, D> = (Vec<TensorMap<R, D>>, Vec<TensorMap<R, D, CudaStorage<D>>>);

fn cuda_chain_tensors_at<R, D>(
    runtime: &Runtime,
    space: &GradedSpace<R>,
    seed: u64,
) -> TypedChainPair<R, D>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync,
    D: tenet::typed::CudaPayload,
{
    let host: Vec<_> = (0..3)
        .map(|index| {
            TensorMap::<R, D>::rand_with_seed(runtime, [space], [space], seed + index).unwrap()
        })
        .collect();
    let device = host.iter().map(|t| t.to_cuda().unwrap()).collect();
    (host, device)
}

/// Leaf C2 (#1336), extended in C4 (#1341) to a non-Abelian provider: a
/// `tensor!` device chain at `f32` and `Complex32` produces the host result of
/// the same dtype, cold and warm, and the warm replay keeps the
/// retained-destination contract of the `f64` chain.
#[test]
#[ignore = "requires a real CUDA device"]
fn single_precision_device_chains_match_the_host_and_reuse_their_destinations() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let u1 = GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 4), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    // SU(2): several coupled sectors carrying different quantum dimensions, so
    // the chain's recoupling — not only its GEMMs — runs at single precision.
    let su2 = GradedSpace::try_new_with_arc(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 3),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();

    /// `K * sqrt(terms) * eps(real(D))` at the chain's own scale: derived from
    /// the payload's epsilon, never an absolute platform constant.
    fn chain_tolerance(epsilon: f64) -> f64 {
        32.0 * (64.0_f64).sqrt() * epsilon
    }

    fn assert_chain<R, D>(runtime: &Runtime, space: &GradedSpace<R>, seed: u64, tolerance: f64)
    where
        R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
            + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec
            + Send
            + Sync,
        D: tenet::typed::CudaPayload + Copy + std::fmt::Debug,
    {
        let (host, device) = cuda_chain_tensors_at::<R, D>(runtime, space, seed);
        let oracle = host[0]
            .contract(&host[1], &[1], &[0], &[0, 1])
            .unwrap()
            .contract(&host[2], &[1], &[0], &[0, 1])
            .unwrap();
        let cold =
            tensor!([a; d] = (device[0])[a; b] * (device[1])[b; c] * (device[2])[c; d]).unwrap();
        let warm =
            tensor!([a; d] = (device[0])[a; b] * (device[1])[b; c] * (device[2])[c; d]).unwrap();
        for (label, result) in [("cold", cold), ("warm", warm)] {
            let actual = result.to_host().unwrap();
            assert_eq!(actual.data().len(), oracle.data().len());
            for (index, (&got, &want)) in actual.data().iter().zip(oracle.data()).enumerate() {
                let distance = (got.widen_complex() - want.widen_complex()).norm();
                assert!(
                    distance <= tolerance * (1.0 + want.widen_complex().norm()),
                    "{label} chain element {index}: {got:?} vs {want:?} (distance {distance})"
                );
            }
        }
    }

    let double = chain_tolerance(f64::EPSILON);
    let single = chain_tolerance(f64::from(f32::EPSILON));

    assert_chain::<_, f64>(&runtime, &u1, 762_100, double);
    assert_chain::<_, f32>(&runtime, &u1, 762_200, single);
    assert_chain::<_, Complex64>(&runtime, &u1, 762_300, double);
    assert_chain::<_, Complex32>(&runtime, &u1, 762_400, single);

    assert_chain::<_, f64>(&runtime, &su2, 762_500, double);
    assert_chain::<_, f32>(&runtime, &su2, 762_600, single);
    assert_chain::<_, Complex64>(&runtime, &su2, 762_700, double);
    assert_chain::<_, Complex32>(&runtime, &su2, 762_800, single);
}

/// Leaf C2 (#1336): the warm single-precision chain costs the same device
/// calls as the `f64` chain of the same fixture and moves half the bytes.
///
/// Relative, same-process comparison: both dtypes run the same fixture in this
/// binary, so no absolute platform constant appears. The counters are
/// process-wide, so this test needs `--test-threads=1`.
#[test]
#[ignore = "requires a real CUDA device"]
fn a_warm_single_precision_chain_costs_the_same_calls_and_half_the_bytes() {
    use tenet::dense::cuda_transfer_stats;

    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let u1 = GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 4), (U1Irrep::new(1), 2)],
    )
    .unwrap();

    fn warm_cost<D>(
        runtime: &Runtime,
        u1: &GradedSpace<U1FusionRule>,
        seed: u64,
    ) -> (u64, u64, u64, u64, u64, u64)
    where
        D: tenet::typed::CudaPayload,
    {
        let (_, device) = cuda_chain_tensors_at::<U1FusionRule, D>(runtime, u1, seed);
        let run =
            || tensor!([a; d] = (device[0])[a; b] * (device[1])[b; c] * (device[2])[c; d]).unwrap();
        // Two warm-up runs: the first has nothing retained, the second is the
        // first to overwrite retained destinations.
        drop(run());
        drop(run());
        let before = cuda_transfer_stats();
        drop(run());
        let after = cuda_transfer_stats();
        (
            after.h2d_calls - before.h2d_calls,
            after.h2d_bytes - before.h2d_bytes,
            after.d2h_calls - before.d2h_calls,
            after.device_allocs - before.device_allocs,
            after.copy_calls - before.copy_calls,
            after.gemm_calls - before.gemm_calls,
        )
    }

    // Both real and complex lanes: the complex twin is what shows the halving
    // is the *element size*, not the real/complex split (#1341).
    for (name, single, double) in [
        (
            "f32/f64",
            warm_cost::<f32>(&runtime, &u1, 763_200),
            warm_cost::<f64>(&runtime, &u1, 763_100),
        ),
        (
            "c32/c64",
            warm_cost::<Complex32>(&runtime, &u1, 763_400),
            warm_cost::<Complex64>(&runtime, &u1, 763_300),
        ),
    ] {
        eprintln!("warm chain {name}: narrow {single:?} wide {double:?}");
        assert_eq!(
            (single.0, single.2, single.3, single.4, single.5),
            (double.0, double.2, double.3, double.4, double.5),
            "{name}: (h2d_calls, d2h_calls, device_allocs, copy_calls, gemm_calls) must not \
             depend on the dtype"
        );
        assert!(double.1 > 0, "{name}: vacuous byte count");
        assert_eq!(
            single.1 * 2,
            double.1,
            "{name}: warm h2d bytes must be halved"
        );
    }
}

// ---------------------------------------------------------------------------
// General device networks (G2c-3, #1348)
// ---------------------------------------------------------------------------

/// Device result against a reference of the same dtype, compared at the
/// payload's own epsilon (never bitwise: the device GEMM order is cuTENSOR's
/// and exact zeros may differ in sign).
fn assert_close_dyn<D: tenet::typed::CudaPayload + std::fmt::Debug>(
    actual: &[D],
    expected: &[D],
    epsilon: f64,
    what: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{what}: payload length");
    let scale = expected
        .iter()
        .map(|value| value.widen_complex().norm())
        .fold(0.0_f64, f64::max);
    assert!(scale > 0.0, "{what}: all-zero reference proves nothing");
    let tolerance = 256.0 * epsilon * (1.0 + scale);
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        let distance = (got.widen_complex() - want.widen_complex()).norm();
        assert!(
            distance <= tolerance,
            "{what}: element {index} is {got:?}, expected {want:?} (tolerance {tolerance:e})"
        );
    }
}

/// Runs one `tensor!` expression over `$t` bound to the Host operands, then
/// twice — cold and warm — over `$t` bound to the device operands.
macro_rules! host_cold_warm {
    ($t:ident = $host:expr, $device:expr; $($net:tt)*) => {{
        let host = {
            let $t = $host;
            tensor!($($net)*).unwrap()
        };
        let (cold, warm) = {
            let $t = $device;
            (tensor!($($net)*).unwrap(), tensor!($($net)*).unwrap())
        };
        (host, cold, warm)
    }};
}

/// The operands of the general networks, over three spaces `v`, `w`, `p` of
/// one provider (degeneracy > 1 in each):
///
/// - `a: [v, w; p]`, `b: [p, w; v, p]`, `c: [v; w]`, `e: [w; w]`, and `t` with
///   `conj(t)` shaped like `b` — codomain↔domain contractions only, so the
///   physical-basis dense expansion is a plain index sum;
/// - `f: [v, w; p]`, `g: [v*, w; p*]`, `h: [w*, v;]` — codomain↔codomain and
///   domain↔domain pairs, i.e. dual contracted legs (the fermionic twist).
struct Operands<T> {
    a: T,
    b: T,
    c: T,
    e: T,
    t: T,
    f: T,
    g: T,
    h: T,
}

impl<T> Operands<T> {
    fn map<U>(&self, lift: impl Fn(&T) -> U) -> Operands<U> {
        Operands {
            a: lift(&self.a),
            b: lift(&self.b),
            c: lift(&self.c),
            e: lift(&self.e),
            t: lift(&self.t),
            f: lift(&self.f),
            g: lift(&self.g),
            h: lift(&self.h),
        }
    }
}

fn general_operands<R, D>(
    runtime: &Runtime,
    [v, w, p]: [&GradedSpace<R>; 3],
    seed: u64,
) -> Operands<TensorMap<R, D>>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: tenet::typed::CudaPayload,
{
    let rand = |codomain: &[&GradedSpace<R>], domain: &[&GradedSpace<R>], salt: u64| {
        TensorMap::<R, D>::rand_with_seed(
            runtime,
            codomain.iter().copied(),
            domain.iter().copied(),
            seed + salt,
        )
        .unwrap()
    };
    let (v_dual, w_dual, p_dual) = (
        v.try_dual().unwrap(),
        w.try_dual().unwrap(),
        p.try_dual().unwrap(),
    );
    Operands {
        a: rand(&[v, w], &[p], 0),
        b: rand(&[p, w], &[v, p], 1),
        c: rand(&[v], &[w], 2),
        e: rand(&[w], &[w], 3),
        t: rand(&[v, p], &[p, w], 4),
        f: rand(&[v, w], &[p], 5),
        g: rand(&[&v_dual, w], &[&p_dual], 6),
        h: rand(&[&w_dual, v], &[], 7),
    }
}

/// Every general network on Host and device (cold, warm): general contraction
/// axes, open outputs on both sides, result and final permutations (a
/// split-changing one included), a lazy conjugate operand, dual contracted
/// legs, and a four-tensor chain whose two intermediates are reused warm.
/// Returns `(name, host, device cold, device warm)`.
#[allow(clippy::type_complexity)]
fn general_networks<R, D>(
    host: &Operands<TensorMap<R, D>>,
    device: &Operands<TensorMap<R, D, CudaStorage<D>>>,
) -> Vec<(
    &'static str,
    TensorMap<R, D>,
    TensorMap<R, D, CudaStorage<D>>,
    TensorMap<R, D, CudaStorage<D>>,
)>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync,
    D: tenet::typed::CudaPayload + Send + Sync + 'static,
{
    let mut runs = Vec::new();
    let (host_result, cold, warm) = host_cold_warm!(o = host, device;
        [x, q; y] = (o.a)[a, w; p] * (o.b)[p, x; a, y] * (o.c)[q; w]);
    runs.push(("xq;y", host_result, cold, warm));
    let (host_result, cold, warm) = host_cold_warm!(o = host, device;
        [q, x; y] = (o.a)[a, w; p] * (o.b)[p, x; a, y] * (o.c)[q; w]);
    runs.push(("qx;y", host_result, cold, warm));
    let (host_result, cold, warm) = host_cold_warm!(o = host, device;
        [y, q; x] = (o.a)[a, w; p] * (o.b)[p, x; a, y] * (o.c)[q; w]);
    runs.push(("yq;x split-changing", host_result, cold, warm));
    let (host_result, cold, warm) = host_cold_warm!(o = host, device;
        [x, q; y] = (o.a)[a, w; p] * conj((o.t))[a, y; p, x] * (o.c)[q; w]);
    runs.push(("xq;y lazy conj", host_result, cold, warm));
    let (host_result, cold, warm) = host_cold_warm!(o = host, device;
        [x, q; y] = (o.a)[a, w; p] * (o.b)[p, x; a, y] * (o.c)[q; u] * (o.e)[u; w]);
    runs.push(("xq;y four tensors", host_result, cold, warm));
    let (host_result, cold, warm) = host_cold_warm!(o = host, device;
        [z, x;] = (o.f)[a, w; p] * (o.g)[a, x; p] * (o.h)[w, z;]);
    runs.push(("zx; dual legs", host_result, cold, warm));
    let (host_result, cold, warm) = host_cold_warm!(o = host, device;
        [x; z] = (o.f)[a, w; p] * (o.g)[a, x; p] * (o.h)[w, z;]);
    runs.push(("x;z dual legs split-changing", host_result, cold, warm));
    runs
}

fn assert_general_networks_match_host<R, D>(
    runtime: &Runtime,
    spaces: [&GradedSpace<R>; 3],
    seed: u64,
    epsilon: f64,
    provider: &str,
) where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync,
    D: tenet::typed::CudaPayload + Send + Sync + 'static + std::fmt::Debug,
{
    let operands = general_operands::<R, D>(runtime, spaces, seed);
    let device = operands.map(|tensor| tensor.to_cuda().unwrap());
    for (name, host, cold, warm) in general_networks(&operands, &device) {
        let what = format!("{provider} {name}");
        for (phase, result) in [("cold", cold), ("warm", warm)] {
            assert_eq!(result.placement(), device.a.placement(), "{what} {phase}");
            assert!(std::ptr::eq(result.provider(), host.provider()), "{what}");
            assert_eq!(result.codomain(), host.codomain(), "{what} {phase}");
            assert_eq!(result.domain(), host.domain(), "{what} {phase}");
            assert_close_dyn(
                result.to_host().unwrap().data(),
                host.data(),
                epsilon,
                &format!("{what} {phase}"),
            );
        }
    }
}

fn column_major_offset(shape: &[usize], index: &[usize]) -> usize {
    index
        .iter()
        .zip(shape)
        .rev()
        .fold(0, |offset, (&i, &extent)| offset * extent + i)
}

/// Physical-basis einsum: every operand `(dense, labels)` in codomain-then-
/// domain axis order, summed over every label not in `output`. Plain index
/// pairing is exact for codomain↔domain contractions and legs that stay on
/// their side, which is all the dense-checked networks use.
fn dense_einsum(
    operands: &[(&tenet::prelude::PhysicalDense<Complex64>, &[&str])],
    output: &[&str],
) -> (Vec<usize>, Vec<Complex64>) {
    let mut labels: Vec<&str> = Vec::new();
    let mut extents = Vec::new();
    for (dense, operand_labels) in operands {
        assert_eq!(dense.shape.len(), operand_labels.len());
        for (&label, &extent) in operand_labels.iter().zip(&dense.shape) {
            match labels.iter().position(|&known| known == label) {
                Some(position) => assert_eq!(extents[position], extent, "label {label}"),
                None => {
                    labels.push(label);
                    extents.push(extent);
                }
            }
        }
    }
    let position = |label: &str| labels.iter().position(|&known| known == label).unwrap();
    let output_shape: Vec<usize> = output
        .iter()
        .map(|&label| extents[position(label)])
        .collect();
    let mut result = vec![Complex64::new(0.0, 0.0); output_shape.iter().product()];
    let total: usize = extents.iter().product();
    let mut index = vec![0usize; labels.len()];
    for _ in 0..total {
        let mut product = Complex64::new(1.0, 0.0);
        for (dense, operand_labels) in operands {
            let local: Vec<usize> = operand_labels.iter().map(|&l| index[position(l)]).collect();
            product *= dense.data[column_major_offset(&dense.shape, &local)];
        }
        let out: Vec<usize> = output.iter().map(|&l| index[position(l)]).collect();
        result[column_major_offset(&output_shape, &out)] += product;
        for (axis, value) in index.iter_mut().enumerate() {
            *value += 1;
            if *value < extents[axis] {
                break;
            }
            *value = 0;
        }
    }
    (output_shape, result)
}

fn widened(dense: tenet::prelude::PhysicalDense<f64>) -> tenet::prelude::PhysicalDense<Complex64> {
    tenet::prelude::PhysicalDense {
        shape: dense.shape,
        data: dense
            .data
            .into_iter()
            .map(|x| Complex64::new(x, 0.0))
            .collect(),
    }
}

/// Dense-expansion oracle for the codomain↔domain networks on a provider with
/// a physical basis: the Host result and the device results, cold and warm,
/// each equal the physical-basis einsum of the operands.
fn assert_dense_oracle<R>(
    runtime: &Runtime,
    spaces: [&GradedSpace<R>; 3],
    seed: u64,
    provider: &str,
) where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + tenet::core::PhysicalFusionBasis<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync,
{
    let o = general_operands::<R, f64>(runtime, spaces, seed);
    let device = o.map(|tensor| tensor.to_cuda().unwrap());
    // Host, device cold and device warm results of one expression, each to be
    // checked against the einsum independently.
    type Runs<R> = (
        TensorMap<R, f64>,
        TensorMap<R, f64, CudaStorage>,
        TensorMap<R, f64, CudaStorage>,
    );
    let results =
        |(host, cold, warm): Runs<R>| [host, cold.to_host().unwrap(), warm.to_host().unwrap()];
    let dense = |tensor: &TensorMap<R, f64>| widened(tensor.to_physical_dense().unwrap());
    let (a, b, c, e) = (dense(&o.a), dense(&o.b), dense(&o.c), dense(&o.e));
    let t_adjoint = dense(&o.t.adjoint().unwrap());
    type DenseOperands<'a> = Vec<(&'a tenet::prelude::PhysicalDense<Complex64>, &'a [&'a str])>;
    type DenseCase<'a, R> = (
        &'a str,
        [TensorMap<R, f64>; 3],
        DenseOperands<'a>,
        &'a [&'a str],
    );
    let cases: [DenseCase<'_, R>; 4] = [
        (
            "xq;y",
            results(
                host_cold_warm!(o = &o, &device; [x, q; y] = (o.a)[a, w; p] * (o.b)[p, x; a, y] * (o.c)[q; w]),
            ),
            vec![
                (&a, &["a", "w", "p"][..]),
                (&b, &["p", "x", "a", "y"][..]),
                (&c, &["q", "w"][..]),
            ],
            &["x", "q", "y"],
        ),
        (
            "qx;y",
            results(
                host_cold_warm!(o = &o, &device; [q, x; y] = (o.a)[a, w; p] * (o.b)[p, x; a, y] * (o.c)[q; w]),
            ),
            vec![
                (&a, &["a", "w", "p"][..]),
                (&b, &["p", "x", "a", "y"][..]),
                (&c, &["q", "w"][..]),
            ],
            &["q", "x", "y"],
        ),
        (
            "xq;y lazy conj",
            results(
                host_cold_warm!(o = &o, &device; [x, q; y] = (o.a)[a, w; p] * conj((o.t))[a, y; p, x] * (o.c)[q; w]),
            ),
            vec![
                (&a, &["a", "w", "p"][..]),
                (&t_adjoint, &["p", "x", "a", "y"][..]),
                (&c, &["q", "w"][..]),
            ],
            &["x", "q", "y"],
        ),
        (
            "xq;y four tensors",
            results(
                host_cold_warm!(o = &o, &device; [x, q; y] = (o.a)[a, w; p] * (o.b)[p, x; a, y] * (o.c)[q; u] * (o.e)[u; w]),
            ),
            vec![
                (&a, &["a", "w", "p"][..]),
                (&b, &["p", "x", "a", "y"][..]),
                (&c, &["q", "u"][..]),
                (&e, &["u", "w"][..]),
            ],
            &["x", "q", "y"],
        ),
    ];
    for (name, runs, operands, output) in cases {
        let (shape, expected) = dense_einsum(&operands, output);
        for (phase, result) in ["host", "device cold", "device warm"]
            .into_iter()
            .zip(&runs)
        {
            let actual = dense(result);
            assert_eq!(
                actual.shape, shape,
                "{provider} {name} {phase}: dense shape"
            );
            assert_close_dyn(
                &actual.data,
                &expected,
                f64::EPSILON,
                &format!("{provider} {name} {phase} dense"),
            );
        }
    }
}

fn u1_space(sectors: &[(i32, usize)]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        sectors
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

fn su2_space(sectors: &[(usize, usize)]) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(SU2FusionRule),
        sectors
            .iter()
            .map(|&(twice, degeneracy)| (SU2Irrep::from_twice_spin(twice), degeneracy)),
    )
    .unwrap()
}

/// G2c-3 (#1348): general `tensor!` networks on device equal the Host run of
/// the same expression, cold and warm, for U(1), SU(2), U(1)×SU(2),
/// fZ2×U(1) and fZ2⊠SU(2) at `f64` and `Complex64` (U(1) also at the single
/// precision payloads); the codomain↔domain networks of U(1) and SU(2) also
/// equal the physical-basis dense expansion.
#[test]
#[ignore = "requires a real CUDA device"]
fn general_cuda_networks_match_host_and_dense_oracles() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();

    let (v, w, p) = (
        u1_space(&[(-1, 2), (0, 1), (1, 2)]),
        u1_space(&[(0, 2), (1, 1)]),
        u1_space(&[(-1, 1), (0, 2), (1, 1)]),
    );
    assert_general_networks_match_host::<_, f64>(
        &runtime,
        [&v, &w, &p],
        790_000,
        f64::EPSILON,
        "U(1)",
    );
    assert_general_networks_match_host::<_, Complex64>(
        &runtime,
        [&v, &w, &p],
        790_100,
        f64::EPSILON,
        "U(1)",
    );
    assert_general_networks_match_host::<_, f32>(
        &runtime,
        [&v, &w, &p],
        790_200,
        f64::from(f32::EPSILON),
        "U(1)",
    );
    assert_general_networks_match_host::<_, Complex32>(
        &runtime,
        [&v, &w, &p],
        790_300,
        f64::from(f32::EPSILON),
        "U(1)",
    );
    assert_dense_oracle(&runtime, [&v, &w, &p], 790_400, "U(1)");

    let (v, w, p) = (
        su2_space(&[(0, 2), (1, 1)]),
        su2_space(&[(1, 2), (2, 1)]),
        su2_space(&[(0, 1), (1, 2)]),
    );
    assert_general_networks_match_host::<_, f64>(
        &runtime,
        [&v, &w, &p],
        791_000,
        f64::EPSILON,
        "SU(2)",
    );
    assert_general_networks_match_host::<_, Complex64>(
        &runtime,
        [&v, &w, &p],
        791_100,
        f64::EPSILON,
        "SU(2)",
    );
    assert_dense_oracle(&runtime, [&v, &w, &p], 791_200, "SU(2)");

    let u1_su2 = Arc::new(U1FusionRule.product(SU2FusionRule));
    let product = |sectors: &[(i32, usize, usize)]| {
        GradedSpace::try_new_with_arc(
            Arc::clone(&u1_su2),
            sectors.iter().map(|&(charge, twice, degeneracy)| {
                (
                    product_sector(U1Irrep::new(charge), SU2Irrep::from_twice_spin(twice)),
                    degeneracy,
                )
            }),
        )
        .unwrap()
    };
    let (v, w, p) = (
        // Charge parity equals twice-spin parity in every sector (the
        // Hubbard-like correlation), so every fusion product meets its
        // partner and no network result is trivially empty.
        product(&[(0, 0, 2), (1, 1, 1)]),
        product(&[(0, 0, 1), (1, 1, 2), (-1, 1, 1)]),
        product(&[(0, 0, 1), (1, 1, 2), (2, 0, 1)]),
    );
    assert_general_networks_match_host::<_, f64>(
        &runtime,
        [&v, &w, &p],
        792_000,
        f64::EPSILON,
        "U(1)xSU(2)",
    );
    assert_general_networks_match_host::<_, Complex64>(
        &runtime,
        [&v, &w, &p],
        792_100,
        f64::EPSILON,
        "U(1)xSU(2)",
    );

    let fz2_u1 = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let fermion_u1 = |sectors: &[(bool, i32, usize)]| {
        GradedSpace::try_new_with_arc(
            Arc::clone(&fz2_u1),
            sectors.iter().map(|&(odd, charge, degeneracy)| {
                (
                    product_sector(
                        if odd { Z2Irrep::ODD } else { Z2Irrep::EVEN },
                        U1Irrep::new(charge),
                    ),
                    degeneracy,
                )
            }),
        )
        .unwrap()
    };
    let (v, w, p) = (
        fermion_u1(&[(false, 0, 2), (true, 1, 1), (true, -1, 1)]),
        fermion_u1(&[(false, 0, 1), (true, 1, 2)]),
        fermion_u1(&[(true, 0, 1), (false, 1, 2), (true, 1, 1)]),
    );
    assert_general_networks_match_host::<_, f64>(
        &runtime,
        [&v, &w, &p],
        793_000,
        f64::EPSILON,
        "fZ2xU(1)",
    );
    assert_general_networks_match_host::<_, Complex64>(
        &runtime,
        [&v, &w, &p],
        793_100,
        f64::EPSILON,
        "fZ2xU(1)",
    );

    let fz2_su2 = Arc::new(FermionParityFusionRule.product(SU2FusionRule));
    let fermion_su2 = |sectors: &[(bool, usize, usize)]| {
        GradedSpace::try_new_with_arc(
            Arc::clone(&fz2_su2),
            sectors.iter().map(|&(odd, twice, degeneracy)| {
                (
                    product_sector(
                        if odd { Z2Irrep::ODD } else { Z2Irrep::EVEN },
                        SU2Irrep::from_twice_spin(twice),
                    ),
                    degeneracy,
                )
            }),
        )
        .unwrap()
    };
    let (v, w, p) = (
        fermion_su2(&[(false, 0, 2), (true, 1, 1)]),
        fermion_su2(&[(true, 1, 2), (false, 2, 1)]),
        fermion_su2(&[(false, 0, 1), (true, 1, 2)]),
    );
    assert_general_networks_match_host::<_, f64>(
        &runtime,
        [&v, &w, &p],
        794_000,
        f64::EPSILON,
        "fZ2xSU(2)",
    );
    assert_general_networks_match_host::<_, Complex64>(
        &runtime,
        [&v, &w, &p],
        794_100,
        f64::EPSILON,
        "fZ2xSU(2)",
    );
}

/// Device state a rejected or warm call must leave unchanged, or change only
/// as its contract states: the plan cache, the process-wide transfer
/// counters, the cuTENSOR plan cache, the contraction scratch and the
/// tree-transform executor.
#[derive(Debug, PartialEq)]
struct DeviceState {
    plans: tenet_network::PlanCacheStats,
    transfers: tenet::dense::CudaTransferStats,
    cutensor: tenet::dense::CudaPlanCacheStats,
    scratch_bytes: usize,
    transforms: tenet::prelude::CudaTreeTransformStats,
}

fn device_state(runtime: &Runtime) -> DeviceState {
    DeviceState {
        plans: plan_cache_stats(runtime),
        transfers: tenet::dense::cuda_transfer_stats(),
        cutensor: runtime.cuda_plan_cache_stats().unwrap().unwrap(),
        scratch_bytes: runtime.cuda_contract_scratch_bytes().unwrap(),
        transforms: runtime.cuda_tree_transform_stats().unwrap(),
    }
}

/// G2c-3 (#1348) warm-cost contract: a warm general network — general axes,
/// result and final permutations, reused intermediates, and for the fermion
/// the dual-leg twist — transfers exactly the returned output's #740 zero
/// upload (one H2D of its bytes) and allocates exactly that output, downloads
/// nothing, misses and evicts no cuTENSOR plan, and grows no scratch and no
/// executor state. Every intermediate step overwrites its retained buffer.
///
/// The counters are process-wide: run with `--test-threads=1`.
#[test]
#[ignore = "requires a real CUDA device"]
fn warm_general_cuda_networks_transfer_only_the_returned_output() {
    fn warm<R, D>(runtime: &Runtime, spaces: [&GradedSpace<R>; 3], seed: u64, what: &str)
    where
        R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
            + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec
            + Send
            + Sync,
        D: tenet::typed::CudaPayload + Send + Sync + 'static,
    {
        let o =
            general_operands::<R, D>(runtime, spaces, seed).map(|tensor| tensor.to_cuda().unwrap());
        let chain = || {
            tensor!([y, q; x] = (o.a)[a, w; p] * (o.b)[p, x; a, y] * (o.c)[q; u] * (o.e)[u; w])
                .unwrap()
        };
        let dual = || tensor!([x; z] = (o.f)[a, w; p] * (o.g)[a, x; p] * (o.h)[w, z;]).unwrap();
        type Run<'a, R, D> = (&'a str, &'a dyn Fn() -> TensorMap<R, D, CudaStorage<D>>);
        let runs: [Run<'_, R, D>; 2] = [("four-tensor chain", &chain), ("dual legs", &dual)];
        for (name, run) in runs {
            // Call 1 has nothing retained; call 2 is the first to overwrite.
            drop(run());
            drop(run());
            let before = device_state(runtime);
            let output = run();
            let after = device_state(runtime);
            let output_bytes = std::mem::size_of_val(output.to_host().unwrap().data()) as u64;
            let what = format!("{what} {name}");
            assert_eq!(
                (
                    after.transfers.h2d_calls - before.transfers.h2d_calls,
                    after.transfers.h2d_bytes - before.transfers.h2d_bytes,
                    after.transfers.d2h_calls - before.transfers.d2h_calls,
                    after.transfers.device_allocs - before.transfers.device_allocs,
                ),
                (1, output_bytes, 0, 1),
                "{what}: (h2d calls, h2d bytes, d2h calls, device allocations)"
            );
            assert_eq!(after.cutensor.misses, before.cutensor.misses, "{what}");
            assert_eq!(
                after.cutensor.evictions, before.cutensor.evictions,
                "{what}"
            );
            assert!(
                after.cutensor.hits > before.cutensor.hits,
                "{what}: vacuous"
            );
            assert_eq!(after.scratch_bytes, before.scratch_bytes, "{what}");
            assert_eq!(after.transforms, before.transforms, "{what}");
            assert_eq!(after.plans.entries, before.plans.entries, "{what}");
            assert_eq!(
                after.plans.workspaces_created, before.plans.workspaces_created,
                "{what}"
            );
        }
    }

    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let (v, w, p) = (
        u1_space(&[(-1, 2), (0, 1), (1, 2)]),
        u1_space(&[(0, 2), (1, 1)]),
        u1_space(&[(-1, 1), (0, 2), (1, 1)]),
    );
    warm::<_, f64>(&runtime, [&v, &w, &p], 795_000, "U(1) f64");
    warm::<_, Complex64>(&runtime, [&v, &w, &p], 795_100, "U(1) c64");
    let fz2_u1 = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let space = |sectors: &[(bool, i32, usize)]| {
        GradedSpace::try_new_with_arc(
            Arc::clone(&fz2_u1),
            sectors.iter().map(|&(odd, charge, degeneracy)| {
                (
                    product_sector(
                        if odd { Z2Irrep::ODD } else { Z2Irrep::EVEN },
                        U1Irrep::new(charge),
                    ),
                    degeneracy,
                )
            }),
        )
        .unwrap()
    };
    let (v, w, p) = (
        space(&[(false, 0, 2), (true, 1, 1), (true, -1, 1)]),
        space(&[(false, 0, 1), (true, 1, 2)]),
        space(&[(true, 0, 1), (false, 1, 2), (true, 1, 1)]),
    );
    warm::<_, f64>(&runtime, [&v, &w, &p], 795_200, "fZ2xU(1) f64");
}

/// G2c-3 (#1348): the device rejection classes reachable through `tensor!`
/// are decided before the plan cache publishes, a workspace is leased or the
/// device is touched: plan cache, pools, transfer counters, cuTENSOR plans,
/// scratch and executor state are all unchanged. (A compact operand is not
/// constructible as a device operand of this impl; its preflight, and the
/// anyonic contraction class, are in the device-free `device_operand_admission`
/// test.) The trace pre-step's rejections (G2c-5), an anyonic operand
/// included, are in
/// `rejected_cuda_trace_prestep_leaves_every_device_state_unchanged`.
#[test]
#[ignore = "requires a real CUDA device"]
fn rejected_cuda_networks_leave_every_device_state_unchanged() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let space = u1_space(&[(0, 2), (1, 1)]);
    let wrong = u1_space(&[(0, 3)]);
    let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&space], [&space], 796_000)
        .unwrap()
        .to_cuda()
        .unwrap();
    let mismatched = TensorMap::<_, f64>::rand_with_seed(&runtime, [&wrong], [&space], 796_001)
        .unwrap()
        .to_cuda()
        .unwrap();
    // Warm the device context so lazily created state is not attributed to
    // the rejected calls.
    drop(tensor!([i; k] = a[i; j] * a[j; k]).unwrap());
    let before = device_state(&runtime);

    // A fresh topology, i.e. the plan-cache miss path only: this pins that the
    // mismatch is raised before a miss publishes. On a topology hit it is
    // raised by the execution body, after the hit is counted and a workspace
    // leased (#1371).
    assert!(tensor!([k; i] = a[i; j] * mismatched[j; k]).is_err());
    assert_eq!(
        device_state(&runtime),
        before,
        "contracted-leg space mismatch"
    );
}

// ---------------------------------------------------------------------------
// Device `tensor!` trace pre-step (G2c-5, #1350)
// ---------------------------------------------------------------------------

/// Traced operands over three spaces `v`, `w`, `p` of one provider:
///
/// - `ta: [v, w; v, p]`, `tb: [p, v; p, w]`, `tt: [v, w; v, w]` — traces
///   between a codomain and a domain leg, so the physical-basis dense
///   expansion traces by a plain diagonal sum;
/// - `td: [v, v*, w; p]` (codomain–codomain) and `te: [p; v, w, w*]`
///   (domain–domain) — dual traced legs;
/// - `c: [v; w]` untraced.
struct TraceOperands<T> {
    ta: T,
    tb: T,
    tt: T,
    td: T,
    te: T,
    c: T,
}

impl<T> TraceOperands<T> {
    fn map<U>(&self, lift: impl Fn(&T) -> U) -> TraceOperands<U> {
        TraceOperands {
            ta: lift(&self.ta),
            tb: lift(&self.tb),
            tt: lift(&self.tt),
            td: lift(&self.td),
            te: lift(&self.te),
            c: lift(&self.c),
        }
    }
}

fn trace_operands<R, D>(
    runtime: &Runtime,
    [v, w, p]: [&GradedSpace<R>; 3],
    seed: u64,
) -> TraceOperands<TensorMap<R, D>>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: tenet::typed::CudaPayload,
{
    let rand = |codomain: &[&GradedSpace<R>], domain: &[&GradedSpace<R>], salt: u64| {
        TensorMap::<R, D>::rand_with_seed(
            runtime,
            codomain.iter().copied(),
            domain.iter().copied(),
            seed + salt,
        )
        .unwrap()
    };
    let (v_dual, w_dual) = (v.try_dual().unwrap(), w.try_dual().unwrap());
    TraceOperands {
        ta: rand(&[v, w], &[v, p], 0),
        tb: rand(&[p, v], &[p, w], 1),
        tt: rand(&[v, w], &[v, w], 2),
        td: rand(&[v, &v_dual, w], &[p], 3),
        te: rand(&[p], &[v, w, &w_dual], 4),
        c: rand(&[v], &[w], 5),
    }
}

/// Every trace-bearing network on Host and device (cold, warm): traces on one,
/// two and three operands, open outputs on both sides, a split-changing final
/// permutation, dual traced legs on either side, a lazy conjugate traced
/// operand, and a full two-pair trace with no contraction step.
#[allow(clippy::type_complexity)]
fn trace_networks<R, D>(
    host: &TraceOperands<TensorMap<R, D>>,
    device: &TraceOperands<TensorMap<R, D, CudaStorage<D>>>,
) -> Vec<(
    &'static str,
    TensorMap<R, D>,
    TensorMap<R, D, CudaStorage<D>>,
    TensorMap<R, D, CudaStorage<D>>,
)>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync,
    D: tenet::typed::CudaPayload + Send + Sync + 'static,
{
    let mut runs = Vec::new();
    let (h, cold, warm) = host_cold_warm!(o = host, device;
        [a; x] = (o.tb)[j, a; j, w] * (o.ta)[i, w; i, x]);
    runs.push(("two traced operands", h, cold, warm));
    let (h, cold, warm) = host_cold_warm!(o = host, device;
        [b; x] = (o.c)[b; w] * (o.ta)[i, w; i, x]);
    runs.push(("traced and untraced", h, cold, warm));
    let (h, cold, warm) = host_cold_warm!(o = host, device;
        [x, a;] = (o.tb)[j, a; j, w] * (o.ta)[i, w; i, x]);
    runs.push(("split-changing final permutation", h, cold, warm));
    let (h, cold, warm) = host_cold_warm!(o = host, device;
        [y; x] = (o.td)[k, k, y; p] * (o.te)[p; x, u, u]);
    runs.push(("dual traced legs", h, cold, warm));
    let (h, cold, warm) = host_cold_warm!(o = host, device;
        [a; x] = (o.tb)[j, a; j, w] * (o.ta)[i, w; i, p] * (o.te)[p; x, u, u]);
    runs.push(("three traced operands", h, cold, warm));
    let (h, cold, warm) = host_cold_warm!(o = host, device;
        [y; x] = conj((o.ta))[i, x; i, p] * (o.ta)[j, y; j, p]);
    runs.push(("lazy conj traced operand", h, cold, warm));
    let (h, cold, warm) = host_cold_warm!(o = host, device; [] = (o.tt)[i, j; i, j]);
    runs.push(("full two-pair trace", h, cold, warm));
    runs
}

fn assert_trace_networks_match_host<R, D>(
    runtime: &Runtime,
    spaces: [&GradedSpace<R>; 3],
    seed: u64,
    epsilon: f64,
    provider: &str,
) where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync,
    D: tenet::typed::CudaPayload + Send + Sync + 'static + std::fmt::Debug,
{
    let operands = trace_operands::<R, D>(runtime, spaces, seed);
    let device = operands.map(|tensor| tensor.to_cuda().unwrap());
    for (name, host, cold, warm) in trace_networks(&operands, &device) {
        let what = format!("{provider} {name}");
        for (phase, result) in [("cold", cold), ("warm", warm)] {
            assert_eq!(result.placement(), device.ta.placement(), "{what} {phase}");
            assert!(std::ptr::eq(result.provider(), host.provider()), "{what}");
            assert_eq!(result.codomain(), host.codomain(), "{what} {phase}");
            assert_eq!(result.domain(), host.domain(), "{what} {phase}");
            assert_close_dyn(
                result.to_host().unwrap().data(),
                host.data(),
                epsilon,
                &format!("{what} {phase}"),
            );
        }
    }
}

/// Dense-expansion oracle for the codomain–domain traces: Host, device cold
/// and device warm each equal the physical-basis einsum, in which a label
/// written twice on one operand is its diagonal sum.
fn assert_trace_dense_oracle<R>(
    runtime: &Runtime,
    spaces: [&GradedSpace<R>; 3],
    seed: u64,
    provider: &str,
) where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + tenet::core::PhysicalFusionBasis<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync,
{
    let o = trace_operands::<R, f64>(runtime, spaces, seed);
    let device = o.map(|tensor| tensor.to_cuda().unwrap());
    type Runs<R> = (
        TensorMap<R, f64>,
        TensorMap<R, f64, CudaStorage>,
        TensorMap<R, f64, CudaStorage>,
    );
    let results =
        |(host, cold, warm): Runs<R>| [host, cold.to_host().unwrap(), warm.to_host().unwrap()];
    let dense = |tensor: &TensorMap<R, f64>| widened(tensor.to_physical_dense().unwrap());
    let (ta, tb, tt, c) = (dense(&o.ta), dense(&o.tb), dense(&o.tt), dense(&o.c));
    let ta_adjoint = dense(&o.ta.adjoint().unwrap());
    type DenseOperands<'a> = Vec<(&'a tenet::prelude::PhysicalDense<Complex64>, &'a [&'a str])>;
    type DenseCase<'a, R> = (
        &'a str,
        [TensorMap<R, f64>; 3],
        DenseOperands<'a>,
        &'a [&'a str],
    );
    let cases: [DenseCase<'_, R>; 4] = [
        (
            "two traced operands",
            results(host_cold_warm!(o = &o, &device;
                [a; x] = (o.tb)[j, a; j, w] * (o.ta)[i, w; i, x])),
            vec![
                (&tb, &["j", "a", "j", "w"][..]),
                (&ta, &["i", "w", "i", "x"][..]),
            ],
            &["a", "x"],
        ),
        (
            "traced and untraced",
            results(host_cold_warm!(o = &o, &device;
                [b; x] = (o.c)[b; w] * (o.ta)[i, w; i, x])),
            vec![(&c, &["b", "w"][..]), (&ta, &["i", "w", "i", "x"][..])],
            &["b", "x"],
        ),
        (
            "lazy conj traced operand",
            results(host_cold_warm!(o = &o, &device;
                [y; x] = conj((o.ta))[i, x; i, p] * (o.ta)[j, y; j, p])),
            vec![
                (&ta_adjoint, &["i", "p", "i", "x"][..]),
                (&ta, &["j", "y", "j", "p"][..]),
            ],
            &["y", "x"],
        ),
        (
            "full two-pair trace",
            results(host_cold_warm!(o = &o, &device; [] = (o.tt)[i, j; i, j])),
            vec![(&tt, &["i", "j", "i", "j"][..])],
            &[],
        ),
    ];
    for (name, runs, operands, output) in cases {
        let (shape, expected) = dense_einsum(&operands, output);
        for (phase, result) in ["host", "device cold", "device warm"]
            .into_iter()
            .zip(&runs)
        {
            let actual = dense(result);
            assert_eq!(
                actual.shape, shape,
                "{provider} {name} {phase}: dense shape"
            );
            assert_close_dyn(
                &actual.data,
                &expected,
                f64::EPSILON,
                &format!("{provider} {name} {phase} dense"),
            );
        }
    }
}

fn fermion_u1_trace_spaces(
) -> [GradedSpace<tenet::core::ProductFusionRule<FermionParityFusionRule, U1FusionRule>>; 3] {
    let rule = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let space = |sectors: &[(bool, i32, usize)]| {
        GradedSpace::try_new_with_arc(
            Arc::clone(&rule),
            sectors.iter().map(|&(odd, charge, degeneracy)| {
                (
                    product_sector(
                        if odd { Z2Irrep::ODD } else { Z2Irrep::EVEN },
                        U1Irrep::new(charge),
                    ),
                    degeneracy,
                )
            }),
        )
        .unwrap()
    };
    // `p` holds the vacuum, so the domain–domain trace of `te` is not empty.
    [
        space(&[(false, 0, 2), (true, 1, 1), (true, -1, 1)]),
        space(&[(false, 0, 1), (true, 1, 2), (true, -1, 1)]),
        space(&[(false, 0, 1), (true, 1, 2), (false, 1, 1)]),
    ]
}

fn fermion_su2_trace_spaces(
) -> [GradedSpace<tenet::core::ProductFusionRule<FermionParityFusionRule, SU2FusionRule>>; 3] {
    let rule = Arc::new(FermionParityFusionRule.product(SU2FusionRule));
    let space = |sectors: &[(bool, usize, usize)]| {
        GradedSpace::try_new_with_arc(
            Arc::clone(&rule),
            sectors.iter().map(|&(odd, twice, degeneracy)| {
                (
                    product_sector(
                        if odd { Z2Irrep::ODD } else { Z2Irrep::EVEN },
                        SU2Irrep::from_twice_spin(twice),
                    ),
                    degeneracy,
                )
            }),
        )
        .unwrap()
    };
    [
        space(&[(false, 0, 2), (true, 1, 1)]),
        space(&[(true, 1, 2), (false, 2, 1)]),
        space(&[(false, 0, 1), (true, 1, 2)]),
    ]
}

/// G2c-5 (#1350): `tensor!` networks with an intra-operand trace pre-step run
/// on device and equal the Host run of the same expression, cold and warm,
/// for U(1) (all four device dtypes), SU(2), fZ2×U(1) and fZ2⊠SU(2) (`f64`
/// and `Complex64`), dual traced legs included; the codomain–domain traces of
/// U(1) and SU(2) also equal the physical-basis dense expansion.
#[test]
#[ignore = "requires a real CUDA device"]
fn trace_prestep_cuda_networks_match_host_and_dense_oracles() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();

    let (v, w, p) = (
        u1_space(&[(-1, 2), (0, 1), (1, 2)]),
        u1_space(&[(0, 2), (1, 1)]),
        u1_space(&[(-1, 1), (0, 2), (1, 1)]),
    );
    let u1 = [&v, &w, &p];
    assert_trace_networks_match_host::<_, f64>(&runtime, u1, 1_350_000, f64::EPSILON, "U(1)");
    assert_trace_networks_match_host::<_, Complex64>(&runtime, u1, 1_350_100, f64::EPSILON, "U(1)");
    let single = f64::from(f32::EPSILON);
    assert_trace_networks_match_host::<_, f32>(&runtime, u1, 1_350_200, single, "U(1)");
    assert_trace_networks_match_host::<_, Complex32>(&runtime, u1, 1_350_300, single, "U(1)");
    assert_trace_dense_oracle(&runtime, u1, 1_350_400, "U(1)");

    let (v, w, p) = (
        su2_space(&[(0, 2), (1, 1)]),
        su2_space(&[(1, 2), (2, 1)]),
        su2_space(&[(0, 1), (1, 2)]),
    );
    let su2 = [&v, &w, &p];
    assert_trace_networks_match_host::<_, f64>(&runtime, su2, 1_351_000, f64::EPSILON, "SU(2)");
    assert_trace_networks_match_host::<_, Complex64>(
        &runtime,
        su2,
        1_351_100,
        f64::EPSILON,
        "SU(2)",
    );
    assert_trace_dense_oracle(&runtime, su2, 1_351_200, "SU(2)");

    let [v, w, p] = fermion_u1_trace_spaces();
    let fu1 = [&v, &w, &p];
    assert_trace_networks_match_host::<_, f64>(&runtime, fu1, 1_352_000, f64::EPSILON, "fZ2xU(1)");
    assert_trace_networks_match_host::<_, Complex64>(
        &runtime,
        fu1,
        1_352_100,
        f64::EPSILON,
        "fZ2xU(1)",
    );

    let [v, w, p] = fermion_su2_trace_spaces();
    let fsu2 = [&v, &w, &p];
    assert_trace_networks_match_host::<_, f64>(
        &runtime,
        fsu2,
        1_353_000,
        f64::EPSILON,
        "fZ2xSU(2)",
    );
    assert_trace_networks_match_host::<_, Complex64>(
        &runtime,
        fsu2,
        1_353_100,
        f64::EPSILON,
        "fZ2xSU(2)",
    );
}

/// G2c-5 (#1350) warm-cost contract: a warm trace-bearing network transfers
/// exactly one #740 zero upload per traced operand (its trace output, which is
/// call-local like the Host's) plus, when the network has a contraction step,
/// the returned output's; it allocates exactly those outputs, downloads
/// nothing, misses and evicts no cuTENSOR plan, and grows no scratch and no
/// executor state. A network without a step returns its trace output itself.
///
/// The counters are process-wide: run with `--test-threads=1`.
#[test]
#[ignore = "requires a real CUDA device"]
fn warm_trace_prestep_transfers_only_the_trace_and_returned_outputs() {
    fn bytes<R, D>(tensor: &TensorMap<R, D>) -> u64
    where
        D: tenet::typed::CudaPayload,
    {
        std::mem::size_of_val(tensor.data()) as u64
    }

    fn warm<R, D>(runtime: &Runtime, spaces: [&GradedSpace<R>; 3], seed: u64, what: &str)
    where
        R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
            + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec
            + Send
            + Sync,
        D: tenet::typed::CudaPayload + Send + Sync + 'static,
    {
        let h = trace_operands::<R, D>(runtime, spaces, seed);
        let o = h.map(|tensor| tensor.to_cuda().unwrap());
        let traced = |tensor: &TensorMap<R, D>, pairs: &[(usize, usize)]| {
            bytes(&tensor.trace_pairs(pairs).unwrap())
        };
        let two = || tensor!([a; x] = (o.tb)[j, a; j, w] * (o.ta)[i, w; i, x]).unwrap();
        let dual = || tensor!([y; x] = (o.td)[k, k, y; p] * (o.te)[p; x, u, u]).unwrap();
        let full = || tensor!([] = (o.tt)[i, j; i, j]).unwrap();
        type Run<'a, R, D> = (
            &'a str,
            &'a dyn Fn() -> TensorMap<R, D, CudaStorage<D>>,
            u64,
            u64,
            bool,
        );
        let runs: [Run<'_, R, D>; 3] = [
            (
                "two traced operands",
                &two,
                2,
                traced(&h.tb, &[(0, 2)]) + traced(&h.ta, &[(0, 2)]),
                true,
            ),
            (
                "dual traced legs",
                &dual,
                2,
                traced(&h.td, &[(0, 1)]) + traced(&h.te, &[(2, 3)]),
                true,
            ),
            (
                "full two-pair trace",
                &full,
                1,
                traced(&h.tt, &[(0, 2), (1, 3)]),
                false,
            ),
        ];
        for (name, run, traces, trace_bytes, steps) in runs {
            drop(run());
            drop(run());
            let before = device_state(runtime);
            let output = run();
            let after = device_state(runtime);
            let output_bytes = bytes(&output.to_host().unwrap());
            let (calls, h2d_bytes) = if steps {
                (traces + 1, trace_bytes + output_bytes)
            } else {
                (traces, trace_bytes)
            };
            let what = format!("{what} {name}");
            assert_eq!(
                (
                    after.transfers.h2d_calls - before.transfers.h2d_calls,
                    after.transfers.h2d_bytes - before.transfers.h2d_bytes,
                    after.transfers.d2h_calls - before.transfers.d2h_calls,
                    after.transfers.device_allocs - before.transfers.device_allocs,
                ),
                (calls, h2d_bytes, 0, calls),
                "{what}: (h2d calls, h2d bytes, d2h calls, device allocations)"
            );
            assert_eq!(after.cutensor.misses, before.cutensor.misses, "{what}");
            assert_eq!(
                after.cutensor.evictions, before.cutensor.evictions,
                "{what}"
            );
            assert!(
                after.cutensor.hits > before.cutensor.hits,
                "{what}: vacuous"
            );
            assert_eq!(after.scratch_bytes, before.scratch_bytes, "{what}");
            assert_eq!(after.transforms, before.transforms, "{what}");
            assert_eq!(after.plans.entries, before.plans.entries, "{what}");
            assert_eq!(after.plans.misses, before.plans.misses, "{what}");
            assert_eq!(
                after.plans.workspaces_created, before.plans.workspaces_created,
                "{what}"
            );
        }
    }

    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let (v, w, p) = (
        u1_space(&[(-1, 2), (0, 1), (1, 2)]),
        u1_space(&[(0, 2), (1, 1)]),
        u1_space(&[(-1, 1), (0, 2), (1, 1)]),
    );
    warm::<_, f64>(&runtime, [&v, &w, &p], 1_354_000, "U(1) f64");
    warm::<_, Complex64>(&runtime, [&v, &w, &p], 1_354_100, "U(1) c64");
    let [v, w, p] = fermion_u1_trace_spaces();
    warm::<_, f64>(&runtime, [&v, &w, &p], 1_354_200, "fZ2xU(1) f64");
}

/// G2c-5 (#1350): every trace of an expression is validated and compiled on
/// the Host before the first one runs, so a trace rejection on a later operand
/// — mutually non-dual traced legs, a label count that is not the operand's
/// rank — leaves plan cache, pools, transfer counters, cuTENSOR plans, scratch
/// and executor state unchanged, with the Host's error. A contracted-leg
/// mismatch between the reduced operands is the Host's own post-trace input
/// error: it is raised after the traces ran (their uploads are its only
/// transfers) and publishes no plan; deciding it before the traces and the
/// plan lookup is #1371. An anyonic traced operand is rejected by the trace
/// compile, with the Host's error, before any trace runs.
#[test]
#[ignore = "requires a real CUDA device"]
fn rejected_cuda_trace_prestep_leaves_every_device_state_unchanged() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let (v, w, p) = (
        u1_space(&[(-1, 2), (0, 1), (1, 2)]),
        u1_space(&[(0, 2), (1, 1)]),
        u1_space(&[(-1, 1), (0, 2), (1, 1)]),
    );
    let h = trace_operands::<_, f64>(&runtime, [&v, &w, &p], 1_355_000);
    let o = h.map(|tensor| tensor.to_cuda().unwrap());
    let host_bad =
        TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &w], [&p, &w], 1_355_100).unwrap();
    let bad = host_bad.to_cuda().unwrap();
    let host_mismatch =
        TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &p], [&v, &w], 1_355_200).unwrap();
    let mismatch = host_mismatch.to_cuda().unwrap();
    drop(tensor!([a; x] = (o.tb)[j, a; j, w] * (o.ta)[i, w; i, x]).unwrap());
    let before = device_state(&runtime);

    let host_error = tensor!([a; x] = (h.tb)[j, a; j, w] * host_bad[i, w; i, x]).unwrap_err();
    let device_error = tensor!([a; x] = (o.tb)[j, a; j, w] * bad[i, w; i, x]).unwrap_err();
    assert_eq!(device_error.to_string(), host_error.to_string());
    assert_eq!(device_state(&runtime), before, "non-dual traced legs");

    let host_error = tensor!([a; x] = (h.tb)[j, a; j, w] * (h.c)[i, w; i, x]).unwrap_err();
    let device_error = tensor!([a; x] = (o.tb)[j, a; j, w] * (o.c)[i, w; i, x]).unwrap_err();
    assert!(matches!(
        device_error,
        tenet::prelude::Error::InvalidArgument(_)
    ));
    assert_eq!(device_error.to_string(), host_error.to_string());
    assert_eq!(
        device_state(&runtime),
        before,
        "label count is not the rank"
    );

    assert!(tensor!([a; x] = (h.tb)[j, a; j, w] * host_mismatch[i, w; i, x]).is_err());
    let before = device_state(&runtime);
    assert!(tensor!([a; x] = (o.tb)[j, a; j, w] * mismatch[i, w; i, x]).is_err());
    let after = device_state(&runtime);
    // This pins only that no plan is published; the hit counted and the
    // workspace quarantined by the failed execution are #1371.
    assert_eq!(
        (
            after.plans.entries,
            after.plans.misses,
            after.plans.topology_materializations
        ),
        (
            before.plans.entries,
            before.plans.misses,
            before.plans.topology_materializations
        ),
        "reduced contracted-leg mismatch publishes no plan"
    );
    assert_eq!(
        after.transfers.h2d_calls - before.transfers.h2d_calls,
        2,
        "reduced contracted-leg mismatch: exactly the two trace outputs"
    );
    assert_eq!(after.transfers.d2h_calls, before.transfers.d2h_calls);
}

/// A one-sector real rule that reports anyonic braiding (every symbol is 1):
/// a valid device operand, so the anyonic boundary is reachable on device.
struct RealAnyonicProbe;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct AnyonicProbeSector;

impl tenet::core::FusionRule for RealAnyonicProbe {
    fn rule_identity(&self) -> tenet::core::RuleIdentity {
        tenet::core::RuleIdentity::from_canonical_bytes::<Self>(
            0x1350_0000_0000_0001,
            Arc::<[u8]>::from([]),
        )
    }
    fn fusion_style(&self) -> tenet::core::FusionStyleKind {
        tenet::core::FusionStyleKind::Unique
    }
    fn braiding_style(&self) -> tenet::core::BraidingStyleKind {
        tenet::core::BraidingStyleKind::Anyonic
    }
    fn vacuum(&self) -> tenet::core::SectorId {
        tenet::core::SectorId::new(0)
    }
    fn fusion_channels(
        &self,
        _: tenet::core::SectorId,
        _: tenet::core::SectorId,
    ) -> tenet::core::SectorVec {
        core::iter::once(tenet::core::SectorId::new(0)).collect()
    }
}

impl tenet::core::MultiplicityFreeFusionRule for RealAnyonicProbe {}

impl tenet::core::MultiplicityFreeFusionSymbols for RealAnyonicProbe {
    type Scalar = f64;
    fn f_symbol_scalar(
        &self,
        _: tenet::core::SectorId,
        _: tenet::core::SectorId,
        _: tenet::core::SectorId,
        _: tenet::core::SectorId,
        _: tenet::core::SectorId,
        _: tenet::core::SectorId,
    ) -> f64 {
        1.0
    }
    fn r_symbol_scalar(
        &self,
        _: tenet::core::SectorId,
        _: tenet::core::SectorId,
        _: tenet::core::SectorId,
    ) -> f64 {
        1.0
    }
}

impl MultiplicityFreeRigidSymbols for RealAnyonicProbe {
    fn dim_scalar(&self, _: tenet::core::SectorId) -> f64 {
        1.0
    }
    fn inv_dim_scalar(&self, _: tenet::core::SectorId) -> f64 {
        1.0
    }
    fn sqrt_dim_scalar(&self, _: tenet::core::SectorId) -> f64 {
        1.0
    }
    fn inv_sqrt_dim_scalar(&self, _: tenet::core::SectorId) -> f64 {
        1.0
    }
    fn twist_scalar(&self, _: tenet::core::SectorId) -> f64 {
        1.0
    }
    fn frobenius_schur_phase_scalar(&self, _: tenet::core::SectorId) -> f64 {
        1.0
    }
}

impl CheckedFusionAlgebra for RealAnyonicProbe {
    fn try_dual_sector(
        &self,
        sector: tenet::core::SectorId,
    ) -> Result<tenet::core::SectorId, FusionAlgebraError> {
        Ok(sector)
    }
    fn try_fusion_channels(
        &self,
        left: tenet::core::SectorId,
        right: tenet::core::SectorId,
    ) -> Result<tenet::core::SectorVec, FusionAlgebraError> {
        Ok(tenet::core::FusionRule::fusion_channels(self, left, right))
    }
    fn try_nsymbol(
        &self,
        left: tenet::core::SectorId,
        right: tenet::core::SectorId,
        coupled: tenet::core::SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        Ok(tenet::core::FusionRule::nsymbol(self, left, right, coupled))
    }
}

impl SectorCodec for RealAnyonicProbe {
    type Sector = AnyonicProbeSector;
    fn encode_sector(
        &self,
        _: &AnyonicProbeSector,
    ) -> Result<tenet::core::SectorId, FusionAlgebraError> {
        Ok(tenet::core::SectorId::new(0))
    }
    fn decode_sector(
        &self,
        sector: tenet::core::SectorId,
    ) -> Result<AnyonicProbeSector, FusionAlgebraError> {
        if sector == tenet::core::SectorId::new(0) {
            Ok(AnyonicProbeSector)
        } else {
            Err(FusionAlgebraError::InvalidSector { sector })
        }
    }
}

/// G2c-5 (#1350): an anyonic traced operand is rejected by the trace compile
/// with the Host's error before any trace runs, leaving every device state
/// unchanged.
#[test]
#[ignore = "requires a real CUDA device"]
fn anyonic_cuda_trace_prestep_rejects_like_host_before_device_work() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(RealAnyonicProbe, [(AnyonicProbeSector, 2)]).unwrap();
    let host = TensorMap::<_, f64>::rand_with_seed(&runtime, [&leg], [&leg], 1_356_000).unwrap();
    let device = host.to_cuda().unwrap();
    let host_error = tensor!([] = host[i; i]).unwrap_err();
    let before = device_state(&runtime);
    let device_error = tensor!([] = device[i; i]).unwrap_err();
    assert!(
        matches!(
            &device_error,
            tenet::prelude::Error::Operation(operation)
                if matches!(
                    **operation,
                    tenet::operations::OperationError::UnsupportedTensorContractScope { .. }
                )
        ),
        "{device_error:?}"
    );
    assert_eq!(device_error.to_string(), host_error.to_string());
    assert_eq!(device_state(&runtime), before, "anyonic traced operand");
}
