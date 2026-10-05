#![allow(deprecated)]
//! Warm Host compose workspace and compatibility-wrapper allocation contract:
//! after one call at
//! a fixed `B`, `execute` and `execute_into` allocate nothing of TeNeT's on
//! the caller thread. The remaining allocations are the backend's, and every
//! grouped submission pays them, eager included (BLAS provider, #2013):
//! - Tenferro's grouped-GEMM validation, `tenferro-tensor 0.7.1
//!   src/backend.rs:670` (`validate_grouped_gemm`);
//! - the BLAS grouped dispatch's batch list, `tenferro-cpu 0.7.1
//!   src/gemm/mod.rs:1630` (`grouped_gemm_blas_typed`).
//!
//! Its own binary, because it installs a counting global allocator.

mod common;
#[path = "../../tests/support/numerics.rs"]
mod numerics;
mod prepared;

use std::time::Instant;

#[allow(unused_imports)]
use num_complex::{Complex32, Complex64};
use tenet::typed::{ComposePlan, PreparedCompose, Runtime, StackedTensorMap};

use prepared::{filled, members, u1_legs};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

fn allocations(f: impl FnOnce()) -> (usize, usize, std::time::Duration) {
    let (elapsed, allocs) = counting_alloc::measure(|| {
        let started = Instant::now();
        f();
        started.elapsed()
    });
    (allocs.calls as usize, allocs.bytes as usize, elapsed)
}

#[test]
fn warm_host_calls_allocate_only_the_backend_grouped_dispatch() {
    // What: at a fixed B, a second `execute` and a second `execute_into`
    // (whose plan has inactive destination blocks to zero-fill) reuse the
    // handle's output, job list and fill strides, and the Runtime's pooled
    // context. The zero fill adds nothing (`execute_into` == `execute`), and
    // both stay within the per-submission allocations of Tenferro's grouped
    // validator and BLAS dispatch. The cold call's count is the control that shows the
    // counter sees this thread's allocations.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let (v, w) = u1_legs();
    let count = 5;
    let lhs =
        StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&v, &v], &[&w], count, 1)).unwrap();
    let rhs = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&w], &[&v], count, 2)).unwrap();
    let mut dst = StackedTensorMap::pack(&filled::<_, f64>(
        &runtime,
        &[&v, &v],
        &[&v],
        count,
        f64::NAN,
    ))
    .unwrap();
    let mut handle = PreparedCompose::new(&lhs, &rhs).unwrap();

    let cold = allocations(|| {
        handle.execute(&lhs, &rhs).unwrap();
    });
    assert!(
        cold.0 > 0,
        "the cold call allocates its output and job list"
    );
    handle.execute_into(&lhs, &rhs, &mut dst).unwrap();

    let warm = allocations(|| {
        handle.execute(&lhs, &rhs).unwrap();
    });
    let warm_into = allocations(|| {
        handle.execute_into(&lhs, &rhs, &mut dst).unwrap();
    });
    // Upper bound: the two backend allocations named in the module docs.
    assert!(warm.0 <= 2, "warm execute allocations: {warm:?}");
    assert_eq!(
        (warm_into.0, warm_into.1),
        (warm.0, warm.1),
        "the zero fill of execute_into allocates nothing"
    );
    assert!(cold.0 > warm.0 + 2, "cold {cold:?} against warm {warm:?}");

    let plan_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let plan_lhs = StackedTensorMap::pack(&members::<_, f64>(
        &plan_runtime,
        &[&v, &v],
        &[&w],
        count,
        1,
    ))
    .unwrap();
    let plan_rhs =
        StackedTensorMap::pack(&members::<_, f64>(&plan_runtime, &[&w], &[&v], count, 2)).unwrap();
    let plan = ComposePlan::new(&plan_lhs, &plan_rhs).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let plan_cold = allocations(|| {
        plan.execute(&plan_lhs, &plan_rhs, &mut workspace).unwrap();
    });
    let plan_warm = allocations(|| {
        plan.execute(&plan_lhs, &plan_rhs, &mut workspace).unwrap();
    });
    assert_eq!((plan_warm.0, plan_warm.1), (warm.0, warm.1));
    eprintln!(
        "B={count} legacy cold={cold:?} warm={warm:?}; plan/workspace cold={plan_cold:?} warm={plan_warm:?}"
    );
}
