//! Device gates of `PreparedEighFull` (#1499, leaf L3 of #1287).
//!
//! Its own binary, with every test serialized, because `cuda_transfer_stats`
//! and the plan-cache statistics are process- and context-wide. Run with
//! `cargo test -p tenet-rs --no-default-features --features cuda,cpu-faer
//! --test prepared_eigh_cuda -- --ignored`.
//!
//! The device keeps the raw cuSOLVER gauge, and with `B > 1` the batched
//! solver may differ from eager's in the last ULPs, so eigenvectors are
//! checked through gauge-invariant quantities against device eager, the Host
//! eager decomposition and the Jacobi oracle. At `B = 1` the handle must be
//! bit-identical (IEEE `==`) to device eager.

#![cfg(feature = "cuda")]

mod common;
#[path = "../../tests/support/numerics.rs"]
mod numerics;
mod prepared;

use std::collections::BTreeSet;
use std::fmt::Debug;
use std::sync::Mutex;

use num_complex::{Complex32, Complex64};
use tenet::core::{SU2FusionRule, SU2Irrep};
use tenet::dense::{cuda_transfer_stats, CudaPlanCacheStats, CudaTransferStats};
use tenet::prelude::Error;
use tenet::typed::{
    BatchError, GradedSpace, MemberFault, PreparedEighFull, Runtime, StackedTensorMap, TensorMap,
};

use common::DeviceRule;
use prepared::eigh::{
    check_member, degenerate_entry, has_degenerate_group, has_plus_minus_tie, hermitian_members,
    plus_minus_entry, sector_matrices, single_leg,
};
use prepared::{fz2u1_legs, members, su2_legs, u1_legs};

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn delta(after: CudaTransferStats, before: CudaTransferStats) -> CudaTransferStats {
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

fn plans(runtime: &Runtime) -> CudaPlanCacheStats {
    runtime.cuda_plan_cache_stats().unwrap().unwrap()
}

/// Checks every member of one device execute against device eager (and at
/// `B = 1` bit for bit), and against Host eager through the oracle.
fn check_device_batch<R>(label: &str, inputs: &[TensorMap<R, f64>])
where
    R: DeviceRule,
    R::Sector: Debug,
{
    let _ = (Complex32::new(0.0, 0.0), Complex64::new(0.0, 0.0));
    let device: Vec<_> = inputs.iter().map(|t| t.to_cuda().unwrap()).collect();
    let stack = StackedTensorMap::pack(inputs).unwrap().to_cuda().unwrap();
    let mut handle = PreparedEighFull::new(&stack).unwrap();
    let output = handle.execute(&stack).unwrap();
    let d_stack = output.d.to_host().unwrap();
    let v_stack = output.v.to_host().unwrap();
    for (member, (input, device_input)) in inputs.iter().zip(&device).enumerate() {
        let what = format!("{label} member {member}/{}", inputs.len());
        let (eager_d, eager_v) = device_input.eigh_full().unwrap();
        let (eager_d, eager_v) = (eager_d.to_host().unwrap(), eager_v.to_host().unwrap());
        let d = d_stack.member(member).unwrap();
        let v = v_stack.member(member).unwrap();
        assert!(
            d.structure_signature() == eager_d.structure_signature(),
            "{what}: d space"
        );
        assert!(
            v.structure_signature() == eager_v.structure_signature(),
            "{what}: v space"
        );
        if inputs.len() == 1 {
            assert!(
                d.data() == eager_d.data(),
                "{what}: d bit-identical to eager"
            );
            assert!(
                v.data() == eager_v.data(),
                "{what}: v bit-identical to eager"
            );
        }
        // Values in `spectra` are the diagonal of `d`, sector for sector.
        let diagonal = sector_matrices(&d);
        for entry in &output.spectra[member] {
            let matrix = &diagonal[&format!("{:?}", entry.sector)];
            let values: Vec<f64> = (0..matrix.rows)
                .map(|i| matrix.data[i + matrix.rows * i])
                .collect();
            assert!(values == entry.values, "{what}: spectra {:?}", entry.sector);
        }
        check_member(&what, input, &d, &v, (&eager_d, &eager_v), f64::EPSILON);
        let (host_d, host_v) = input.eigh_full().unwrap();
        check_member(
            &format!("{what} vs Host"),
            input,
            &d,
            &v,
            (&host_d, &host_v),
            f64::EPSILON,
        );
    }
}

fn equivalence<R>(label: &str, leg: GradedSpace<R>)
where
    R: DeviceRule,
    R::Sector: Debug,
{
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    for count in [1, 2, 17] {
        let inputs = hermitian_members(&runtime, &[&leg, &leg], count, 3);
        assert!(inputs[0].block_count() > 3, "{label}: several blocks");
        check_device_batch(&format!("{label} B={count}"), &inputs);
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_members_equal_device_eager_and_the_oracle() {
    let _serial = serial();
    equivalence("u1", u1_legs().0);
    equivalence("su2", su2_legs().0);
    equivalence("fz2u1", fz2u1_legs().0);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn plus_minus_lambda_and_degenerate_groups_compare_by_value() {
    let _serial = serial();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let leg = u1_legs().0;
    let j = SU2Irrep::from_twice_spin;
    let su2 = GradedSpace::try_new(SU2FusionRule, [(j(0), 3), (j(1), 4)]).unwrap();
    for count in [1, 17] {
        let pm = single_leg(&runtime, &leg, count, plus_minus_entry);
        assert!(has_plus_minus_tie(&pm[0].eigh_full().unwrap().0));
        check_device_batch(&format!("u1 ±λ B={count}"), &pm);
        let degenerate = single_leg(&runtime, &su2, count, degenerate_entry);
        assert!(has_degenerate_group(&degenerate[0].eigh_full().unwrap().0));
        check_device_batch(&format!("su2 degenerate B={count}"), &degenerate);
    }
}

/// Per call, independent of `B`: one solver call per coupled sector, one
/// gather plus one copy per coupled sector (every route of this fixture is
/// layout-aligned), no GEMM, the same downloads and uploads, and warm calls
/// miss and evict no plan. The ledger reservation counts the contraction
/// plans of the selector and admission steps, which is none: no step
/// submits a contraction, so even the cold call misses no contraction plan.
#[test]
#[ignore = "requires a real CUDA device"]
fn submissions_transfers_and_the_ledger_do_not_depend_on_b() {
    let _serial = serial();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let (leg, _) = su2_legs();
    let mut per_b = BTreeSet::new();
    for count in [1, 2, 17] {
        let inputs = hermitian_members(&runtime, &[&leg, &leg], count, 7);
        let sectors = sector_matrices(&inputs[0]).len() as u64;
        let stack = StackedTensorMap::pack(&inputs).unwrap().to_cuda().unwrap();
        let reserved = plans(&runtime).reserved_entries;
        let mut handle = PreparedEighFull::new(&stack).unwrap();
        assert_eq!(
            plans(&runtime).reserved_entries,
            reserved,
            "no ledger reservation"
        );

        let cold_plans = plans(&runtime);
        let before = cuda_transfer_stats();
        handle.execute(&stack).unwrap();
        let cold = delta(cuda_transfer_stats(), before);
        assert_eq!(
            plans(&runtime).misses,
            cold_plans.misses,
            "B={count}: no contraction plan"
        );

        let warm_plans = plans(&runtime);
        let before = cuda_transfer_stats();
        handle.execute(&stack).unwrap();
        let warm = delta(cuda_transfer_stats(), before);
        let after = plans(&runtime);
        assert_eq!(
            after.misses, warm_plans.misses,
            "B={count}: warm plan misses"
        );
        assert_eq!(
            after.evictions, warm_plans.evictions,
            "B={count}: warm evictions"
        );
        assert_eq!(warm.solver_calls, sectors, "B={count}: solver calls");
        assert_eq!(warm.gemm_calls, 0, "B={count}: no GEMM");
        assert_eq!(
            warm.copy_calls,
            2 * sectors,
            "B={count}: gathers and copies"
        );
        assert_eq!(
            cold.h2d_calls,
            warm.h2d_calls + 1,
            "B={count}: cold allocates v once"
        );
        // Exactly symmetric inputs are decided at stage 2 of admission:
        // two admission downloads plus one spectra download.
        assert_eq!(warm.d2h_calls, 3, "B={count}: downloads");
        per_b.insert((
            warm.h2d_calls,
            warm.d2h_calls,
            warm.copy_calls,
            warm.solver_calls,
        ));
    }
    assert_eq!(
        per_b.len(),
        1,
        "per-call counts independent of B: {per_b:?}"
    );
}

fn rejected(result: Result<(), BatchError>) -> Vec<(usize, MemberFault)> {
    match result {
        Ok(()) => Vec::new(),
        Err(BatchError::MemberRejected { members }) => members,
        Err(other) => panic!("unexpected batch error {other:?}"),
    }
}

/// Non-Hermitian members are all named, the solver never launches, and
/// admission downloads three times whatever `B` is.
#[test]
#[ignore = "requires a real CUDA device"]
fn non_hermitian_members_are_named_before_any_solver_launch() {
    let _serial = serial();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let (leg, _) = u1_legs();
    for count in [3, 17] {
        let mut inputs = hermitian_members(&runtime, &[&leg, &leg], count, 1);
        let x = members::<_, f64>(&runtime, &[&leg, &leg], &[&leg, &leg], 1, 4).remove(0);
        inputs[1] = x.clone();
        inputs[count - 1] = x;
        let stack = StackedTensorMap::pack(&inputs).unwrap().to_cuda().unwrap();
        let mut handle = PreparedEighFull::new(&stack).unwrap();
        let before = cuda_transfer_stats();
        let got = rejected(handle.execute(&stack).map(|_| ()));
        let counts = delta(cuda_transfer_stats(), before);
        let fault = MemberFault::NotHermitian;
        assert_eq!(got, vec![(1, fault), (count - 1, fault)]);
        assert_eq!(counts.solver_calls, 0, "no solver launch");
        assert_eq!(counts.gemm_calls, 0, "no assembly");
        assert_eq!(
            counts.d2h_calls, 3,
            "three admission downloads at B={count}"
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn mixed_scale_admission_verdicts_equal_device_eager_per_member() {
    let _serial = serial();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let (leg, _) = u1_legs();
    let h = hermitian_members(&runtime, &[&leg, &leg], 1, 5).remove(0);
    let x = members::<_, f64>(&runtime, &[&leg, &leg], &[&leg, &leg], 1, 9).remove(0);
    let skewed = h.add(&x, 1.0, 2f64.powi(-30)).unwrap();
    let admitted = h.add(&x, 1.0, 2f64.powi(-60)).unwrap();
    let mut inputs = Vec::new();
    for exponent in [0, -1000, 1000, -1060] {
        let s = 2f64.powi(exponent);
        inputs.extend([h.scale(s), skewed.scale(s), admitted.scale(s)]);
    }
    let expected: Vec<_> = inputs
        .iter()
        .enumerate()
        .filter(|(_, input)| input.to_cuda().unwrap().eigh_full().is_err())
        .map(|(member, _)| (member, MemberFault::NotHermitian))
        .collect();
    assert!(
        expected.len() >= 3 && expected.len() < inputs.len(),
        "{expected:?}"
    );
    let stack = StackedTensorMap::pack(&inputs).unwrap().to_cuda().unwrap();
    let mut handle = PreparedEighFull::new(&stack).unwrap();
    assert_eq!(rejected(handle.execute(&stack).map(|_| ())), expected);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn non_finite_entries_and_eigenvalues_reject_their_members() {
    let _serial = serial();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let (leg, _) = u1_legs();
    let mut inputs = single_leg(&runtime, &leg, 5, |_, _, _| 1.0);
    inputs[1] = inputs[1].scale(f64::NAN);
    let stack = StackedTensorMap::pack(&inputs).unwrap().to_cuda().unwrap();
    let mut handle = PreparedEighFull::new(&stack).unwrap();
    assert_eq!(
        rejected(handle.execute(&stack).map(|_| ())),
        vec![(1, MemberFault::NotHermitian)]
    );

    // Finite, admitted, but `2M` overflows in the solver.
    let huge = f64::MAX / 1.5;
    let mut inputs = single_leg(&runtime, &leg, 5, |_, _, _| 1.0);
    inputs[0] = inputs[0].scale(huge);
    inputs[3] = inputs[3].scale(huge);
    let eager: Vec<bool> = inputs
        .iter()
        .map(|input| input.to_cuda().unwrap().eigh_full().is_err())
        .collect();
    assert_eq!(
        eager,
        [true, false, false, true, false],
        "eager rejects the overflow"
    );
    let stack = StackedTensorMap::pack(&inputs).unwrap().to_cuda().unwrap();
    let mut handle = PreparedEighFull::new(&stack).unwrap();
    let fault = MemberFault::NonFiniteEigenvalue;
    assert_eq!(
        rejected(handle.execute(&stack).map(|_| ())),
        vec![(0, fault), (3, fault)]
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn complex_device_payloads_are_unsupported() {
    let _serial = serial();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let (leg, _) = u1_legs();
    let complex = members::<_, Complex64>(&runtime, &[&leg], &[&leg], 2, 1);
    let stack = StackedTensorMap::pack(&complex).unwrap().to_cuda().unwrap();
    assert!(matches!(
        PreparedEighFull::new(&stack),
        Err(Error::Operation(error)) if format!("{error:?}").contains("real payloads")
    ));
}
