//! Device gates of `PreparedCompose` (#1498, leaf L2 of #1287).
//!
//! Its own binary, with every test serialized, because `cuda_transfer_stats`
//! and the plan-cache statistics are process- and context-wide. Run with
//! `cargo test -p tenet-rs --no-default-features --features cuda,cpu-faer
//! --test prepared_compose_cuda -- --ignored`.

#![cfg(feature = "cuda")]

mod common;
#[path = "../../tests/support/numerics.rs"]
mod numerics;
mod prepared;

use std::collections::{BTreeSet, HashSet};
use std::fmt::Debug;
use std::sync::Mutex;

use num_complex::{Complex32, Complex64};
use tenet::core::U1FusionRule;
use tenet::dense::{cuda_transfer_stats, CudaPlanCacheStats, CudaTransferStats};
use tenet::typed::{GradedSpace, PreparedCompose, Runtime, StackedTensorMap, TensorMap};

use common::{DevicePayload, DeviceRule};
use prepared::{
    assert_close, compose_oracle, expected_plan_entries, filled, fz2u1_legs, members, su2_legs,
    u1_legs,
};

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

/// Coupled sectors both operands carry: the plan's direct GEMM jobs, read
/// from the operands' public block trees rather than from the plan.
fn active_sectors<R, D>(a: &TensorMap<R, D>, b: &TensorMap<R, D>) -> usize
where
    R: DeviceRule,
    R::Sector: Debug,
    D: DevicePayload,
{
    let coupled = |t: &TensorMap<R, D>| {
        (0..t.subblock_count())
            .map(|i| format!("{:?}", t.subblock_fusion_trees(i).unwrap().coupled()))
            .collect::<HashSet<_>>()
    };
    coupled(a).intersection(&coupled(b)).count()
}

fn device_gate<R, D>(label: &str, (v, w): (GradedSpace<R>, GradedSpace<R>))
where
    R: DeviceRule,
    R::Sector: Debug,
    D: DevicePayload,
{
    let _ = (Complex32::new(0.0, 0.0), Complex64::new(0.0, 0.0));
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let mut fills_per_b = BTreeSet::new();
    for count in [1, 2, 17] {
        let label = format!("{label} {} B={count}", D::NAME);
        let a = members::<R, D>(&runtime, &[&v, &v], &[&w], count, 1);
        let b = members::<R, D>(&runtime, &[&w], &[&v], count, 2);
        let jobs = active_sectors(&a[0], &b[0]);
        let lhs = StackedTensorMap::pack(&a).unwrap().to_cuda().unwrap();
        let rhs = StackedTensorMap::pack(&b).unwrap().to_cuda().unwrap();
        let mut handle = PreparedCompose::new(&lhs, &rhs).unwrap();

        // `terms` = len(A), an upper bound on the inner dimension of a block.
        let terms = a[0].data().len();
        let oracles: Vec<_> = a
            .iter()
            .zip(&b)
            .map(|(x, y)| {
                let eager = x.compose(y).unwrap();
                let (oracle, unreached) = compose_oracle(x, y, &eager);
                assert!(unreached > 0, "{label}: fixture must have inactive blocks");
                assert_close(
                    eager.data(),
                    &oracle,
                    terms,
                    &format!("{label}: Host eager"),
                );
                oracle
            })
            .collect();
        let check = |stack: &StackedTensorMap<R, D, tenet::typed::CudaStorage<D>>, what: &str| {
            let host = stack.to_host().unwrap();
            for (index, oracle) in oracles.iter().enumerate() {
                assert_close(
                    host.member(index).unwrap().data(),
                    oracle,
                    terms,
                    &format!("{label}: {what}, member {index}"),
                );
            }
        };

        handle.execute(&lhs, &rhs).unwrap();
        let retained = handle.retained_bytes();
        let (before, plans_before) = (cuda_transfer_stats(), plans(&runtime));
        let output = handle.execute(&lhs, &rhs).unwrap();
        let warm = delta(cuda_transfer_stats(), before);
        let plans_after = plans(&runtime);
        assert_eq!(
            warm.gemm_calls, jobs as u64,
            "{label}: one GEMM per direct job"
        );
        assert_eq!(
            (warm.h2d_calls, warm.d2h_calls),
            (0, 0),
            "{label}: {warm:?}"
        );
        assert_eq!(warm.device_allocs, 0, "{label}: {warm:?}");
        assert_eq!(
            plans_after.misses, plans_before.misses,
            "{label}: warm misses"
        );
        assert_eq!(
            plans_after.evictions, plans_before.evictions,
            "{label}: warm evictions"
        );
        check(output, "execute");
        assert_eq!(
            handle.retained_bytes(),
            retained,
            "{label}: warm retained bytes"
        );

        for (fill, name) in [
            (D::entry(f64::NAN, f64::NAN), "NaN"),
            (D::entry(-3.0e17, 7.5), "garbage"),
        ] {
            let poisoned = || {
                StackedTensorMap::pack(&filled::<R, D>(&runtime, &[&v, &v], &[&v], count, fill))
                    .unwrap()
                    .to_cuda()
                    .unwrap()
            };
            let mut dst = poisoned();
            handle.execute_into(&lhs, &rhs, &mut dst).unwrap();
            check(&dst, &format!("cold execute_into over {name}"));

            let mut dst = poisoned();
            let (before, plans_before) = (cuda_transfer_stats(), plans(&runtime));
            handle.execute_into(&lhs, &rhs, &mut dst).unwrap();
            let warm = delta(cuda_transfer_stats(), before);
            let plans_after = plans(&runtime);
            assert_eq!(
                (warm.h2d_calls, warm.d2h_calls, warm.device_allocs),
                (0, 0, 0),
                "{label}: warm execute_into transfers nothing: {warm:?}"
            );
            assert_eq!(
                plans_after.misses, plans_before.misses,
                "{label}: into misses"
            );
            assert_eq!(
                plans_after.evictions, plans_before.evictions,
                "{label}: into evictions"
            );
            assert!(
                warm.gemm_calls > jobs as u64,
                "{label}: inactive blocks are filled"
            );
            fills_per_b.insert(warm.gemm_calls - jobs as u64);
            check(&dst, &format!("warm execute_into over {name}"));
        }
    }
    assert_eq!(
        fills_per_b.len(),
        1,
        "{label}: zero fills per call must not depend on B: {fills_per_b:?}"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_handle_equals_the_oracle_with_b_independent_submissions() {
    let _guard = serial();
    device_gate::<_, f64>("U1", u1_legs());
    device_gate::<_, Complex64>("U1", u1_legs());
    device_gate::<_, f64>("SU2", su2_legs());
    device_gate::<_, Complex64>("SU2", su2_legs());
    device_gate::<_, f64>("fZ2xU1", fz2u1_legs());
    device_gate::<_, Complex64>("fZ2xU1", fz2u1_legs());
}

fn device<R: DeviceRule>(
    members: &[TensorMap<R, f64>],
) -> StackedTensorMap<R, f64, tenet::typed::CudaStorage<f64>> {
    StackedTensorMap::pack(members).unwrap().to_cuda().unwrap()
}

/// `V ⊗ W ← V ⊗ W` with 81 blocks of distinct extents (the #1508 fixture):
/// its device permute alone needs more than Tenferro's default 64 plans.
fn wide_permute_source(runtime: &Runtime) -> TensorMap<U1FusionRule, f64> {
    let u1 = |charges: Vec<(i32, usize)>| {
        GradedSpace::try_new(
            U1FusionRule,
            charges
                .into_iter()
                .map(|(charge, degeneracy)| (tenet::core::U1Irrep::new(charge), degeneracy)),
        )
        .unwrap()
    };
    let v = u1((0..9).map(|a| (a, a as usize + 1)).collect());
    let w = u1((0..9).map(|j| (100 * j, j as usize + 1)).collect());
    let source = members::<_, f64>(runtime, &[&v, &w], &[&v, &w], 1, 7).remove(0);
    assert_eq!(source.subblock_count(), 81);
    source
}

#[test]
#[ignore = "requires a real CUDA device"]
fn handles_and_an_eager_permute_past_the_default_bound_evict_no_plan() {
    // The pinned order of design §6: both handles reserve and run first, then
    // an eager permute whose executor total alone exceeds 64, then a warm
    // interleave of all three. An absolute raise by any consumer would let
    // one absorb another's reservation and evict here.
    let _guard = serial();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let count = 16;
    let (u1_v, u1_w) = u1_legs();
    let (su2_v, su2_w) = su2_legs();
    let u1_a = members::<_, f64>(&runtime, &[&u1_v, &u1_v], &[&u1_w], count, 1);
    let u1_b = members::<_, f64>(&runtime, &[&u1_w], &[&u1_v], count, 2);
    let su2_a = members::<_, f64>(&runtime, &[&su2_v, &su2_v], &[&su2_w], count, 3);
    let su2_b = members::<_, f64>(&runtime, &[&su2_w], &[&su2_v], count, 4);
    let (u1_lhs, u1_rhs) = (device(&u1_a), device(&u1_b));
    let (su2_lhs, su2_rhs) = (device(&su2_a), device(&su2_b));
    let u1_expected =
        expected_plan_entries(&u1_a[0], &u1_b[0], &u1_a[0].compose(&u1_b[0]).unwrap());
    let su2_expected =
        expected_plan_entries(&su2_a[0], &su2_b[0], &su2_a[0].compose(&su2_b[0]).unwrap());
    let source = wide_permute_source(&runtime).to_cuda().unwrap();

    let reserved = |runtime: &Runtime| plans(runtime).reserved_entries;
    let r0 = reserved(&runtime);
    let mut h1 = PreparedCompose::new(&u1_lhs, &u1_rhs).unwrap();
    let r1 = reserved(&runtime);
    let mut h2 = PreparedCompose::new(&su2_lhs, &su2_rhs).unwrap();
    let r2 = reserved(&runtime);
    let (h1_entries, h2_entries) = (r1 - r0, r2 - r1);
    assert_eq!(h1_entries, u1_expected, "U(1) handle reservation");
    assert_eq!(h2_entries, su2_expected, "SU(2) handle reservation");
    h1.execute(&u1_lhs, &u1_rhs).unwrap();
    h2.execute(&su2_lhs, &su2_rhs).unwrap();

    let _ = source.permute(&[1, 0], &[3, 2]).unwrap();
    let executor = runtime
        .cuda_tree_transform_stats()
        .unwrap()
        .required_plan_entries;
    assert!(
        executor > 64,
        "the permute alone must exceed the default bound"
    );
    let before = plans(&runtime);
    assert_eq!(
        before.reserved_entries,
        r0 + h1_entries + h2_entries + executor
    );

    for _ in 0..3 {
        h1.execute(&u1_lhs, &u1_rhs).unwrap();
        h2.execute(&su2_lhs, &su2_rhs).unwrap();
        let _ = source.permute(&[1, 0], &[3, 2]).unwrap();
    }
    let after = plans(&runtime);
    assert_eq!(after.evictions, before.evictions, "{before:?} -> {after:?}");
    assert_eq!(after.misses, before.misses, "{before:?} -> {after:?}");
    assert_eq!(after.reserved_entries, before.reserved_entries);

    // `execute_into` across a change of B on one handle: the zero regions
    // and the template are rebuilt for the new B, and the result is still
    // the oracle's.
    for wide in [count, 2 * count + 1] {
        let a = members::<_, f64>(&runtime, &[&u1_v, &u1_v], &[&u1_w], wide, 5);
        let b = members::<_, f64>(&runtime, &[&u1_w], &[&u1_v], wide, 6);
        let mut dst = device(&filled::<_, f64>(
            &runtime,
            &[&u1_v, &u1_v],
            &[&u1_v],
            wide,
            f64::NAN,
        ));
        h1.execute_into(&device(&a), &device(&b), &mut dst).unwrap();
        let host = dst.to_host().unwrap();
        for (index, (x, y)) in a.iter().zip(&b).enumerate() {
            let eager = x.compose(y).unwrap();
            let (oracle, _) = compose_oracle(x, y, &eager);
            assert_close(
                host.member(index).unwrap().data(),
                &oracle,
                x.data().len(),
                &format!("execute_into at B={wide}, member {index}"),
            );
        }
    }
    assert_eq!(
        reserved(&runtime),
        before.reserved_entries,
        "a new B reserves nothing"
    );

    drop(h1);
    assert_eq!(reserved(&runtime), before.reserved_entries - h1_entries);
    drop(h2);
    assert_eq!(reserved(&runtime), r0 + executor);
}
