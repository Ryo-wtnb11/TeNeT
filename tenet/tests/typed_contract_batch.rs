mod common;
#[macro_use]
#[allow(unused_macros)]
mod contract_cases;
mod braiding_probe;

use braiding_probe::{ProbeSector, RealBraidingProbe};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::time::{Duration, Instant};

struct CountingAllocator;
thread_local! {
    static MEASURING: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        MEASURING.with(|enabled| {
            if enabled.get() {
                CALLS.set(CALLS.get() + 1);
                BYTES.set(BYTES.get() + layout.size());
            }
        });
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        MEASURING.with(|enabled| {
            if enabled.get() {
                CALLS.set(CALLS.get() + 1);
                BYTES.set(BYTES.get() + layout.size());
            }
        });
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        MEASURING.with(|enabled| {
            if enabled.get() {
                CALLS.set(CALLS.get() + 1);
                BYTES.set(BYTES.get() + size);
            }
        });
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn measure<T>(f: impl FnOnce() -> T) -> (T, Duration, usize, usize) {
    CALLS.set(0);
    BYTES.set(0);
    MEASURING.set(true);
    let start = Instant::now();
    let value = f();
    let elapsed = start.elapsed();
    MEASURING.set(false);
    (value, elapsed, CALLS.get(), BYTES.get())
}
use contract_cases::{
    assert_close, blas_contract_oracle, candidate_core_probes, dense_oracle, fermion_u1,
    fermionic_twist_roles, fill, poisoned_destination, su2, su2_bent, su2_reordered, u1,
    u1_inactive_cases, u1_non_self_dual, u1_reordered, u1_rhs_identity, Case, Payload,
};
use std::sync::Arc;
use tenet::sector::{
    CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, PhysicalFusionBasis, SectorCodec,
};
use tenet::typed::GradedSpace;
use tenet::typed::{ContractPlan, ContractSpec, Error, Runtime, StackedTensorMap, TensorMap};

fn check<R, D>(case: Case<R, D>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + PhysicalFusionBasis<Scalar = f64>,
    D: Payload,
{
    for count in [1, 2, 17] {
        let members: Vec<_> = (0..count)
            .map(|i| {
                let scale = D::entry(1.0 + i as f64 / 8.0, i as f64 / 16.0);
                let lhs = case.lhs.scale(scale);
                let rhs = case.rhs.scale(D::entry(1.0 - i as f64 / 32.0, 0.0));
                Case {
                    name: case.name,
                    lhs,
                    rhs,
                    lhs_axes: case.lhs_axes.clone(),
                    rhs_axes: case.rhs_axes.clone(),
                    output_axes: case.output_axes.clone(),
                    dense: case.dense,
                }
            })
            .collect();
        let left: Vec<_> = members.iter().map(|member| &member.lhs).collect();
        let right: Vec<_> = members.iter().map(|member| &member.rhs).collect();
        let lhs = StackedTensorMap::pack(&left).unwrap();
        let rhs = StackedTensorMap::pack(&right).unwrap();
        let plan = ContractPlan::new(&lhs, &rhs, &case.spec()).unwrap();
        let mut workspace = plan.workspace();
        let result = plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        for (i, member_case) in members.iter().enumerate() {
            let member = result.member(i).unwrap();
            let reference = blas_contract_oracle(member_case);
            assert_close(
                member.dense_data().unwrap(),
                reference.dense_data().unwrap(),
                case.terms(),
                case.name,
            );
            if member_case.dense {
                let (_, expected) = dense_oracle(member_case);
                let physical = member.to_physical_dense().unwrap();
                assert_close(&physical.data, &expected, case.terms(), case.name);
            }
            let eager = member_case.host();
            assert_close(
                member.dense_data().unwrap(),
                eager.dense_data().unwrap(),
                case.terms(),
                case.name,
            );
        }
        let poison: Vec<_> = members.iter().map(poisoned_destination).collect();
        let mut destination = StackedTensorMap::pack(&poison).unwrap();
        plan.execute_into(&lhs, &rhs, &mut destination, &mut workspace)
            .unwrap();
        for (i, member_case) in members.iter().enumerate() {
            let member = destination.member(i).unwrap();
            let reference = blas_contract_oracle(member_case);
            assert_close(
                member.dense_data().unwrap(),
                reference.dense_data().unwrap(),
                case.terms(),
                case.name,
            );
            if member_case.dense {
                let (_, expected) = dense_oracle(member_case);
                assert_close(
                    &member.to_physical_dense().unwrap().data,
                    &expected,
                    case.terms(),
                    case.name,
                );
            }
        }
    }
}

#[test]
fn public_host_batch_matches_reference_steps_and_applicable_physical_basis() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    check(u1_reordered::<f64>(&runtime));
    check(su2_reordered::<f64>(&runtime));
    check(u1_reordered::<tenet::typed::Complex64>(&runtime));
    check(su2_reordered::<tenet::typed::Complex64>(&runtime));
    check(u1_reordered::<f32>(&runtime));
    check(su2_reordered::<tenet::typed::Complex32>(&runtime));
    check(
        u1_inactive_cases::<f64>(&runtime)
            .into_iter()
            .nth(1)
            .unwrap(),
    );
    check(su2_bent::<f64>(&runtime));
    check(u1_rhs_identity::<f64>(&runtime));
}

#[test]
fn public_host_batch_core_and_swapped_core_match_reference_steps() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    fn probes<R, D>(runtime: &Runtime, v: &GradedSpace<R>)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec
            + PhysicalFusionBasis<Scalar = f64>,
        D: Payload,
    {
        for (mut case, _) in candidate_core_probes::<R, D>(runtime, v)
            .into_iter()
            .filter(|(case, _)| matches!(case.name, "C0" | "C2"))
        {
            case.dense = case.name == "C0";
            check(case);
        }
    }
    probes::<_, f64>(&runtime, &u1_non_self_dual());
    probes::<_, tenet::typed::Complex64>(&runtime, &u1_non_self_dual());
    probes::<_, f64>(&runtime, &su2());
    probes::<_, tenet::typed::Complex64>(&runtime, &su2());
    check(
        u1_inactive_cases::<f64>(&runtime)
            .into_iter()
            .next()
            .unwrap(),
    );
}

#[test]
fn swapped_core_uses_rhs_member_length_and_nondefault_output_split() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let v = u1(&[(0, 2), (1, 1)]);
    let w = u1(&[(0, 1), (1, 2)]);
    let x = u1(&[(0, 2), (1, 3)]);
    let lhs = TensorMap::<_, f64>::from_subblock_fn(&runtime, [&v], [&w], fill(171)).unwrap();
    let rhs = TensorMap::<_, f64>::from_subblock_fn(&runtime, [&x, &x], [&v], fill(172)).unwrap();
    let spec = ContractSpec {
        lhs: &[0],
        rhs: &[2],
        codomain: &[1, 2],
        domain: &[0],
    };
    let eager = lhs.contract(&rhs, &spec).unwrap();
    let reference = rhs.compose(&lhs).unwrap();
    assert_close(
        eager.dense_data().unwrap(),
        reference.dense_data().unwrap(),
        64,
        "swapped eager",
    );
    let left = StackedTensorMap::pack(&[&lhs, &lhs]).unwrap();
    let right = StackedTensorMap::pack(&[&rhs, &rhs]).unwrap();
    assert_ne!(
        lhs.dense_data().unwrap().len(),
        rhs.dense_data().unwrap().len()
    );
    let plan = ContractPlan::new(&left, &right, &spec).unwrap();
    let mut workspace = plan.workspace();
    let actual = plan.execute(&left, &right, &mut workspace).unwrap();
    for i in 0..2 {
        assert_eq!(actual.member(i).unwrap().codomain_rank(), 2);
        assert_close(
            actual.member(i).unwrap().dense_data().unwrap(),
            reference.dense_data().unwrap(),
            64,
            "swapped batch",
        );
    }
}

#[test]
fn binding_errors_do_not_write_destination_and_workspaces_are_independent() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    binding_errors_case(&runtime, u1_reordered::<f64>(&runtime));
    let core = candidate_core_probes::<_, f64>(&runtime, &u1_non_self_dual())
        .into_iter()
        .find(|(case, _)| case.name == "C2")
        .unwrap()
        .0;
    binding_errors_case(&runtime, core);
}

fn binding_errors_case(runtime: &Runtime, case: Case<tenet::sector::U1FusionRule, f64>) {
    let lhs = StackedTensorMap::pack(&[&case.lhs, &case.lhs]).unwrap();
    let rhs = StackedTensorMap::pack(&[&case.rhs, &case.rhs]).unwrap();
    let plan = ContractPlan::new(&lhs, &rhs, &case.spec()).unwrap();
    let other_plan = ContractPlan::new(&lhs, &rhs, &case.spec()).unwrap();
    let mut workspace = plan.workspace();
    let mut second = plan.workspace();
    let mut foreign = other_plan.workspace();
    let mut destination =
        StackedTensorMap::pack(&[poisoned_destination(&case), poisoned_destination(&case)])
            .unwrap();
    let bits = |stack: &StackedTensorMap<_, f64>| {
        (0..stack.len())
            .flat_map(|i| {
                stack
                    .member(i)
                    .unwrap()
                    .dense_data()
                    .unwrap()
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let before = bits(&destination);
    let wrong_count = lhs.select(&[0]).unwrap();
    assert!(matches!(
        ContractPlan::new(&wrong_count, &rhs, &case.spec()),
        Err(Error::InvalidArgument(_))
    ));
    assert!(plan
        .execute_into(&wrong_count, &rhs, &mut destination, &mut workspace)
        .is_err());
    assert_eq!(bits(&destination), before);
    assert!(plan
        .execute_into(&lhs, &rhs, &mut destination, &mut foreign)
        .is_err());
    assert_eq!(bits(&destination), before);
    let wrong_destination = destination.select(&[0]).unwrap();
    let mut wrong_destination = wrong_destination;
    let wrong_before = bits(&wrong_destination);
    assert!(plan
        .execute_into(&lhs, &rhs, &mut wrong_destination, &mut workspace)
        .is_err());
    assert_eq!(bits(&wrong_destination), wrong_before);
    let v = u1(&[(0, 1)]);
    let wrong_member = TensorMap::<_, f64>::zeros(runtime, [&v], [&v]).unwrap();
    let wrong = StackedTensorMap::pack(&[wrong_member.clone(), wrong_member]).unwrap();
    assert!(plan
        .execute_into(&wrong, &rhs, &mut destination, &mut workspace)
        .is_err());
    assert_eq!(bits(&destination), before);
    plan.execute(&lhs, &rhs, &mut workspace).unwrap();
    plan.execute(&lhs, &rhs, &mut second).unwrap();
    assert!(workspace.retained_bytes() > 0);
    assert!(workspace.take_output().is_some());
    assert!(workspace.take_output().is_none());
    plan.execute_into(&lhs, &rhs, &mut destination, &mut second)
        .unwrap();
    let singleton = rhs.select(&[0]).unwrap();
    let singleton_lhs = lhs.select(&[0]).unwrap();
    plan.execute(&singleton_lhs, &singleton, &mut workspace)
        .unwrap();
    let many = vec![0; 17];
    let many_lhs = lhs.select(&many).unwrap();
    let many_rhs = rhs.select(&many).unwrap();
    assert_eq!(
        plan.execute(&many_lhs, &many_rhs, &mut workspace)
            .unwrap()
            .len(),
        17
    );
    assert_eq!(plan.execute(&lhs, &rhs, &mut second).unwrap().len(), 2);
}

#[test]
fn unsupported_copy_c_is_explicit() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let v = u1(&[(0, 2), (1, 2)]);
    let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&v, &v], 3).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&v, &v], 4).unwrap();
    let lhs = StackedTensorMap::pack(&[a]).unwrap();
    let rhs = StackedTensorMap::pack(&[b]).unwrap();
    let spec = ContractSpec {
        lhs: &[2, 3],
        rhs: &[0, 1],
        codomain: &[1, 0],
        domain: &[2, 3],
    };
    let result = ContractPlan::new(&lhs, &rhs, &spec);
    assert!(
        matches!(result, Err(Error::Operation(error)) if matches!(*error, tenet::typed::OperationError::UnsupportedTensorContractScope { message: "Host batch contraction does not admit eager copyC output" }))
    );
}

#[test]
fn non_symmetric_braiding_is_rejected_before_plan_compile() {
    let runtime = Runtime::builder().build().unwrap();
    for anyonic in [false, true] {
        let rejected = if anyonic {
            reject_braiding::<true>(&runtime)
        } else {
            reject_braiding::<false>(&runtime)
        };
        assert!(rejected);
    }
}

#[test]
fn twist_bearing_dynamic_tree_is_rejected() {
    let runtime = Runtime::builder().build().unwrap();
    let v = fermion_u1();
    let cases = fermionic_twist_roles::<_, f64>(&runtime, &v, ["A", "canonical", "B", "both"], 71);
    let mut found_twist = false;
    for case in cases {
        let lhs = StackedTensorMap::pack(&[&case.lhs]).unwrap();
        let rhs = StackedTensorMap::pack(&[&case.rhs]).unwrap();
        if matches!(ContractPlan::new(&lhs, &rhs, &case.spec()), Err(Error::Operation(error)) if matches!(*error, tenet::typed::OperationError::UnsupportedTensorContractScope { .. }))
        {
            found_twist = true;
        }
    }
    assert!(found_twist);
}

#[test]
fn fermionic_unit_alpha_core_is_still_rejected() {
    let runtime = Runtime::builder().build().unwrap();
    let case = candidate_core_probes::<_, f64>(&runtime, &fermion_u1())
        .into_iter()
        .find(|(case, _)| case.name == "C0")
        .unwrap()
        .0;
    let lhs = StackedTensorMap::pack(&[&case.lhs]).unwrap();
    let rhs = StackedTensorMap::pack(&[&case.rhs]).unwrap();
    assert!(matches!(ContractPlan::new(&lhs, &rhs, &case.spec()),
        Err(Error::Operation(error)) if matches!(*error,
            tenet::typed::OperationError::UnsupportedTensorContractScope { message }
            if message.contains("bosonic"))));
}

#[test]
fn partitioned_output_uses_requested_codomain_rank() {
    let runtime = Runtime::builder().build().unwrap();
    let case = u1_reordered::<f64>(&runtime);
    let lhs = StackedTensorMap::pack(&[&case.lhs]).unwrap();
    let rhs = StackedTensorMap::pack(&[&case.rhs]).unwrap();
    let mut admitted = 0;
    for split in [0, 1, 3] {
        let (codomain, domain) = case.output_axes.split_at(split);
        let spec = ContractSpec {
            lhs: &case.lhs_axes,
            rhs: &case.rhs_axes,
            codomain,
            domain,
        };
        let Ok(plan) = ContractPlan::new(&lhs, &rhs, &spec) else {
            continue;
        };
        let expected = case.lhs.contract(&case.rhs, &spec).unwrap();
        let actual = plan
            .execute(&lhs, &rhs, &mut plan.workspace())
            .unwrap()
            .member(0)
            .unwrap();
        assert_eq!(actual.codomain_rank(), split);
        assert_close(
            actual.dense_data().unwrap(),
            expected.dense_data().unwrap(),
            case.terms(),
            case.name,
        );
        admitted += 1;
    }
    assert!(admitted > 0, "no non-default output split was admitted");
    let malformed = ContractSpec {
        lhs: &case.lhs_axes,
        rhs: &case.rhs_axes,
        codomain: &[0, 0],
        domain: &[2],
    };
    assert!(ContractPlan::new(&lhs, &rhs, &malformed).is_err());
}

fn reject_braiding<const ANYONIC: bool>(runtime: &Runtime) -> bool {
    let leg =
        GradedSpace::try_new(Arc::new(RealBraidingProbe::<ANYONIC>), [(ProbeSector, 2)]).unwrap();
    let tensor = TensorMap::<_, f64>::zeros(runtime, [&leg], [&leg]).unwrap();
    let stack = StackedTensorMap::pack(&[tensor]).unwrap();
    let spec = ContractSpec {
        lhs: &[1],
        rhs: &[0],
        codomain: &[0],
        domain: &[1],
    };
    matches!(ContractPlan::new(&stack, &stack, &spec), Err(Error::Operation(error)) if matches!(*error, tenet::typed::OperationError::UnsupportedTensorContractScope { message: tenet::typed::NON_SYMMETRIC_CONTRACTION_UNSUPPORTED }))
}

/// Run explicitly with `--release -- --ignored --nocapture`; timing is evidence,
/// never a pass/fail gate. Allocations count the calling thread only.
#[test]
#[ignore]
fn public_host_batch_cold_warm_measurement() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let case = su2_reordered::<f64>(&runtime);
    let (_, eager_time, eager_calls, eager_bytes) =
        measure(|| black_box(case.lhs.contract(&case.rhs, &case.spec()).unwrap()));
    eprintln!("eager B=1 cold: {eager_time:?}, {eager_calls} calls, {eager_bytes} bytes");
    for count in [1, 2, 17] {
        let left = vec![&case.lhs; count];
        let right = vec![&case.rhs; count];
        let lhs = StackedTensorMap::pack(&left).unwrap();
        let rhs = StackedTensorMap::pack(&right).unwrap();
        let ((plan, mut workspace), cold_time, cold_calls, cold_bytes) = measure(|| {
            let plan = ContractPlan::new(&lhs, &rhs, &case.spec()).unwrap();
            let mut workspace = plan.workspace();
            black_box(plan.execute(&lhs, &rhs, &mut workspace).unwrap());
            (plan, workspace)
        });
        let mut samples = Vec::new();
        for _ in 0..31 {
            let (_, elapsed, calls, bytes) = measure(|| {
                black_box(plan.execute(&lhs, &rhs, &mut workspace).unwrap());
            });
            samples.push((elapsed, calls, bytes));
        }
        samples.sort_by_key(|sample| sample.0);
        let (warm_time, warm_calls, warm_bytes) = samples[15];
        eprintln!("B={count}: cold {cold_time:?} {cold_calls} calls {cold_bytes} bytes; warm median {warm_time:?} {warm_calls} calls {warm_bytes} bytes; retained {} bytes", workspace.retained_bytes());
    }
    let mut eager = Vec::new();
    for _ in 0..31 {
        let (_, elapsed, calls, bytes) = measure(|| {
            black_box(case.lhs.contract(&case.rhs, &case.spec()).unwrap());
        });
        eager.push((elapsed, calls, bytes));
    }
    eager.sort_by_key(|sample| sample.0);
    eprintln!(
        "eager B=1 warm median: {:?}, {} calls, {} bytes",
        eager[15].0, eager[15].1, eager[15].2
    );
}

/// Release-only observation of the direct Core/SwappedCore stack path.
#[test]
#[ignore]
fn public_host_swapped_core_release_measurement() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let case = candidate_core_probes::<_, f64>(&runtime, &su2())
        .into_iter()
        .find(|(case, _)| case.name == "C2")
        .unwrap()
        .0;
    for count in [1, 2, 17] {
        let ((lhs, rhs), pack_time, pack_calls, pack_bytes) = measure(|| {
            let left = vec![&case.lhs; count];
            let right = vec![&case.rhs; count];
            (
                StackedTensorMap::pack(&left).unwrap(),
                StackedTensorMap::pack(&right).unwrap(),
            )
        });
        let (plan, plan_time, plan_calls, plan_bytes) =
            measure(|| ContractPlan::new(&lhs, &rhs, &case.spec()).unwrap());
        let (mut workspace, workspace_time, workspace_calls, workspace_bytes) =
            measure(|| plan.workspace());
        let (_, cold_time, cold_calls, cold_bytes) = measure(|| {
            black_box(plan.execute(&lhs, &rhs, &mut workspace).unwrap());
        });
        let mut samples = Vec::new();
        for _ in 0..31 {
            let (_, elapsed, calls, bytes) = measure(|| {
                black_box(plan.execute(&lhs, &rhs, &mut workspace).unwrap());
            });
            samples.push((elapsed, calls, bytes));
        }
        samples.sort_by_key(|sample| sample.0);
        let (warm_time, warm_calls, warm_bytes) = samples[15];
        eprintln!("SwappedCore B={count}: pack {pack_time:?}/{pack_calls}/{pack_bytes}; plan {plan_time:?}/{plan_calls}/{plan_bytes}; workspace {workspace_time:?}/{workspace_calls}/{workspace_bytes}; cold execute {cold_time:?}/{cold_calls}/{cold_bytes}; warm median {warm_time:?}/{warm_calls}/{warm_bytes}; retained {} bytes", workspace.retained_bytes());
    }
    let mut samples = Vec::new();
    for _ in 0..31 {
        let (_, elapsed, calls, bytes) = measure(|| {
            black_box(case.host());
        });
        samples.push((elapsed, calls, bytes));
    }
    samples.sort_by_key(|sample| sample.0);
    eprintln!(
        "eager B=1 warm median: {:?}/{} calls/{} bytes",
        samples[15].0, samples[15].1, samples[15].2
    );
}
