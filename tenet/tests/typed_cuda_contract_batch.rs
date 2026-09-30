//! Public C1/Core and C2/SwappedCore CUDA batch gates. Run on A100 with
//! `cargo test -p tenet-rs --no-default-features --features cuda,cpu-faer
//! --test typed_cuda_contract_batch -- --ignored --test-threads=1`.
#![cfg(feature = "cuda")]

mod braiding_probe;
mod common;
#[macro_use]
#[allow(unused_macros)]
mod contract_cases;

use braiding_probe::{ProbeSector, RealBraidingProbe};
use common::{DevicePayload, DeviceRule};
use contract_cases::{
    assert_close, blas_contract_oracle, candidate_core_probes, fermion_su2, fermion_u1,
    fermionic_blas_contract_oracle_partitioned, fill, poisoned_destination, su2, u1,
    u1_inactive_cases, u1_non_self_dual, Case, TwistRole,
};
use num_complex::Complex64;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tenet::expert::{cuda_transfer_stats, CudaTransferStats};
use tenet::sector::{
    product_sector, FermionParityFusionRule, ProductFusionRuleExt, U1FusionRule, U1Irrep, Z2Irrep,
};
use tenet::typed::{
    ContractPlan, ContractSpec, CudaStorage, Direction, Error, GradedSpace, Runtime,
    StackedTensorMap, TensorMap,
};

fn check_signed_core<R: DeviceRule, D: DevicePayload>(
    runtime: &Runtime,
    space: &GradedSpace<R>,
    swapped: bool,
) {
    let dual = space.try_dual().unwrap();
    let a = TensorMap::<R, D>::from_subblock_fn(runtime, [space], [&dual], fill(201)).unwrap();
    let b = TensorMap::<R, D>::from_subblock_fn(runtime, [&dual], [space], fill(202)).unwrap();
    let base = Case {
        name: "signed CUDA core",
        lhs: if swapped { b.clone() } else { a.clone() },
        rhs: if swapped { a } else { b },
        lhs_axes: vec![usize::from(!swapped)],
        rhs_axes: vec![usize::from(swapped)],
        output_axes: if swapped { vec![1, 0] } else { vec![0, 1] },
        dense: false,
    };
    let first_lhs = StackedTensorMap::pack(&[&base.lhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let first_rhs = StackedTensorMap::pack(&[&base.rhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let plan = ContractPlan::new(&first_lhs, &first_rhs, &base.spec()).unwrap();
    let reserved_before = runtime
        .cuda_plan_cache_stats()
        .unwrap()
        .unwrap()
        .reserved_entries;
    let mut workspace = plan.workspace().unwrap();
    let mut other = plan.workspace().unwrap();
    let mut expected_gemms = None;
    for count in [1, 2, 17] {
        let members: Vec<_> = (0..count)
            .map(|i| Case {
                name: base.name,
                lhs: base
                    .lhs
                    .scale(D::entry(1.0 + i as f64 / 8.0, i as f64 / 16.0)),
                rhs: base.rhs.scale(D::entry(1.0 - i as f64 / 32.0, 0.0)),
                lhs_axes: base.lhs_axes.clone(),
                rhs_axes: base.rhs_axes.clone(),
                output_axes: base.output_axes.clone(),
                dense: false,
            })
            .collect();
        let left: Vec<_> = members.iter().map(|m| &m.lhs).collect();
        let right: Vec<_> = members.iter().map(|m| &m.rhs).collect();
        let lhs = StackedTensorMap::pack(&left).unwrap().to_cuda().unwrap();
        let rhs = StackedTensorMap::pack(&right).unwrap().to_cuda().unwrap();
        let (_, cold) = observe(|| plan.execute(&lhs, &rhs, &mut workspace).unwrap());
        assert!(cold.cuda.h2d_calls >= 1);
        assert_eq!(
            cold.cuda.gemm_calls, 3,
            "three signed structural jobs for this fixture"
        );
        let (_, warm) = observe(|| {
            plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        });
        assert_eq!(warm.cuda.h2d_calls, 0);
        assert_eq!(warm.cuda.d2h_calls, 0);
        let result = plan
            .execute(&lhs, &rhs, &mut workspace)
            .unwrap()
            .to_host()
            .unwrap();
        let gemms = warm.cuda.gemm_calls;
        assert!(gemms >= 3, "warm replay also zeroes inactive blocks");
        if let Some(expected) = expected_gemms {
            assert_eq!(gemms, expected);
        }
        expected_gemms = Some(gemms);
        for (i, member) in members.iter().enumerate() {
            let twist = |tensor: &TensorMap<R, D>, legs: &[usize]| {
                tensor.twist(legs, Direction::Forward).unwrap()
            };
            let role = if swapped { TwistRole::A } else { TwistRole::B };
            let oracle = fermionic_blas_contract_oracle_partitioned(member, role, 1, twist);
            let selected = if swapped {
                Case {
                    name: "selected signed swapped dispatch",
                    lhs: member.rhs.clone(),
                    rhs: member.lhs.clone(),
                    lhs_axes: vec![1],
                    rhs_axes: vec![0],
                    output_axes: vec![0, 1],
                    dense: false,
                }
            } else {
                Case {
                    name: member.name,
                    lhs: member.lhs.clone(),
                    rhs: member.rhs.clone(),
                    lhs_axes: member.lhs_axes.clone(),
                    rhs_axes: member.rhs_axes.clone(),
                    output_axes: member.output_axes.clone(),
                    dense: false,
                }
            };
            let selected_oracle =
                fermionic_blas_contract_oracle_partitioned(&selected, TwistRole::B, 1, twist);
            let untwisted =
                fermionic_blas_contract_oracle_partitioned(&selected, TwistRole::None, 1, twist);
            let expected = oracle.dense_data().unwrap();
            assert_close(
                expected,
                selected_oracle.dense_data().unwrap(),
                member.terms(),
                base.name,
            );
            let selected_data = selected_oracle.dense_data().unwrap();
            let scale = selected_data
                .iter()
                .map(|x| x.magnitude())
                .fold(0.0, f64::max);
            let tolerance = 64.0 * (member.terms().max(1) as f64).sqrt() * D::EPS * (1.0 + scale);
            assert!(
                selected_data
                    .iter()
                    .zip(untwisted.dense_data().unwrap())
                    .any(|(&x, &y)| x.distance(y) > tolerance),
                "twist must change signed {}",
                D::NAME
            );
            assert_close(
                result.member(i).unwrap().dense_data().unwrap(),
                expected,
                member.terms(),
                base.name,
            );
            let eager = member
                .lhs
                .to_cuda()
                .unwrap()
                .contract(&member.rhs.to_cuda().unwrap(), &member.spec())
                .unwrap()
                .to_host()
                .unwrap();
            assert_close(
                result.member(i).unwrap().dense_data().unwrap(),
                eager.dense_data().unwrap(),
                member.terms(),
                base.name,
            );
        }
        let poisoned: Vec<_> = members.iter().map(poisoned_destination).collect();
        let mut dst = StackedTensorMap::pack(&poisoned.iter().collect::<Vec<_>>())
            .unwrap()
            .to_cuda()
            .unwrap();
        let (_, into) = observe(|| plan.execute_into(&lhs, &rhs, &mut dst, &mut other).unwrap());
        assert_eq!(into.cuda.h2d_calls, 0);
        assert_eq!(into.cuda.d2h_calls, 0);
        assert_eq!(into.cuda.gemm_calls, gemms);
        let written = dst.to_host().unwrap();
        for (i, member) in members.iter().enumerate() {
            assert_close(
                written.member(i).unwrap().dense_data().unwrap(),
                result.member(i).unwrap().dense_data().unwrap(),
                member.terms(),
                base.name,
            );
        }
        if count == 2 {
            let before = payload_snapshot(&dst);
            let short_rhs = rhs.select(&[0]).unwrap();
            assert!(plan
                .execute_into(&lhs, &short_rhs, &mut dst, &mut other)
                .is_err());
            assert_eq!(payload_snapshot(&dst), before);
            let mut wrong_dst = dst.select(&[0]).unwrap();
            let wrong_before = payload_snapshot(&wrong_dst);
            assert!(plan
                .execute_into(&lhs, &rhs, &mut wrong_dst, &mut other)
                .is_err());
            assert_eq!(payload_snapshot(&wrong_dst), wrong_before);
            assert_eq!(payload_snapshot(&dst), before);
        }
    }
    drop(plan);
    drop(workspace);
    drop(other);
    assert_eq!(
        runtime
            .cuda_plan_cache_stats()
            .unwrap()
            .unwrap()
            .reserved_entries,
        reserved_before
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn signed_core_and_swapped_core_all_device_dtypes() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    macro_rules! cases {
        ($dtype:ty) => {
            for swapped in [false, true] {
                check_signed_core::<_, $dtype>(&runtime, &fermion_u1(), swapped);
                check_signed_core::<_, $dtype>(&runtime, &fermion_su2(), swapped);
            }
        };
    }
    cases!(f32);
    cases!(num_complex::Complex32);
    cases!(f64);
    cases!(Complex64);
}

#[test]
fn signed_fixture_twist_is_nonvacuous_without_a_device() {
    let runtime = Runtime::builder().build().unwrap();
    for space in [fermion_u1()] {
        let dual = space.try_dual().unwrap();
        let a =
            TensorMap::<_, f64>::from_subblock_fn(&runtime, [&space], [&dual], fill(201)).unwrap();
        let b =
            TensorMap::<_, f64>::from_subblock_fn(&runtime, [&dual], [&space], fill(202)).unwrap();
        for swapped in [false, true] {
            let case = Case {
                name: "signed fixture",
                lhs: if swapped { b.clone() } else { a.clone() },
                rhs: if swapped { a.clone() } else { b.clone() },
                lhs_axes: vec![usize::from(!swapped)],
                rhs_axes: vec![usize::from(swapped)],
                output_axes: if swapped { vec![1, 0] } else { vec![0, 1] },
                dense: false,
            };
            let twist = |tensor: &TensorMap<_, f64>, legs: &[usize]| {
                tensor.twist(legs, Direction::Forward).unwrap()
            };
            let selected = if swapped {
                Case {
                    name: "selected signed swapped dispatch",
                    lhs: case.rhs.clone(),
                    rhs: case.lhs.clone(),
                    lhs_axes: vec![1],
                    rhs_axes: vec![0],
                    output_axes: vec![0, 1],
                    dense: false,
                }
            } else {
                case
            };
            let actual =
                fermionic_blas_contract_oracle_partitioned(&selected, TwistRole::B, 1, twist);
            let absent =
                fermionic_blas_contract_oracle_partitioned(&selected, TwistRole::None, 1, twist);
            assert!(
                actual
                    .dense_data()
                    .unwrap()
                    .iter()
                    .zip(absent.dense_data().unwrap())
                    .any(|(&x, &y)| (x - y).abs() > 1e-10),
                "swapped={swapped}"
            );
        }
    }
    let odd = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule.product(U1FusionRule)),
        [(product_sector(Z2Irrep::ODD, U1Irrep::new(0)), 2)],
    )
    .unwrap();
    let dual = odd.try_dual().unwrap();
    let case = Case {
        name: "negative-only signed core",
        lhs: TensorMap::<_, f64>::from_subblock_fn(&runtime, [&odd], [&dual], |_, _| 1.0).unwrap(),
        rhs: TensorMap::<_, f64>::from_subblock_fn(&runtime, [&dual], [&odd], |_, _| 1.0).unwrap(),
        lhs_axes: vec![1],
        rhs_axes: vec![0],
        output_axes: vec![0, 1],
        dense: false,
    };
    let twist = |tensor: &TensorMap<_, f64>, legs: &[usize]| {
        tensor.twist(legs, Direction::Forward).unwrap()
    };
    let signed = fermionic_blas_contract_oracle_partitioned(&case, TwistRole::B, 1, twist);
    let absent = fermionic_blas_contract_oracle_partitioned(&case, TwistRole::None, 1, twist);
    assert!(signed
        .dense_data()
        .unwrap()
        .iter()
        .zip(absent.dense_data().unwrap())
        .any(|(&x, &y)| (x - y).abs() > 1e-10));
    assert_close(
        case.host().dense_data().unwrap(),
        signed.dense_data().unwrap(),
        case.terms(),
        case.name,
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn signed_core_zeroes_inactive_blocks_after_poisoning() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let rule = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let v = GradedSpace::try_new(
        Arc::clone(&rule),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(-1)), 1),
        ],
    )
    .unwrap();
    let w = GradedSpace::try_new(
        rule,
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
        ],
    )
    .unwrap();
    let wd = w.try_dual().unwrap();
    let case = Case {
        name: "signed inactive",
        lhs: TensorMap::<_, f64>::from_subblock_fn(&runtime, [&v], [&wd], fill(201)).unwrap(),
        rhs: TensorMap::<_, f64>::from_subblock_fn(&runtime, [&wd], [&v], fill(202)).unwrap(),
        lhs_axes: vec![1],
        rhs_axes: vec![0],
        output_axes: vec![0, 1],
        dense: false,
    };
    let oracle = case.host();
    let expected = oracle.dense_data().unwrap();
    assert!(expected.contains(&0.0));
    assert!(expected.iter().any(|&x| x != 0.0));
    let lhs = StackedTensorMap::pack(&[&case.lhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let rhs = StackedTensorMap::pack(&[&case.rhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let plan = ContractPlan::new(&lhs, &rhs, &case.spec()).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let poison = poisoned_destination(&case);
    for count in [1, 2, 17] {
        let lhs = StackedTensorMap::pack(&vec![&case.lhs; count])
            .unwrap()
            .to_cuda()
            .unwrap();
        let rhs = StackedTensorMap::pack(&vec![&case.rhs; count])
            .unwrap()
            .to_cuda()
            .unwrap();
        for _ in 0..2 {
            let mut dst = StackedTensorMap::pack(&vec![&poison; count])
                .unwrap()
                .to_cuda()
                .unwrap();
            plan.execute_into(&lhs, &rhs, &mut dst, &mut workspace)
                .unwrap();
            let result = dst.to_host().unwrap();
            for i in 0..count {
                assert_close(
                    result.member(i).unwrap().dense_data().unwrap(),
                    expected,
                    case.terms(),
                    case.name,
                );
            }
        }
    }
}

fn payload_snapshot<R: DeviceRule, D: DevicePayload>(
    stack: &StackedTensorMap<R, D, CudaStorage<D>>,
) -> String {
    let host = stack.to_host().unwrap();
    format!(
        "{:?}",
        (0..host.len())
            .map(|i| host.member(i).unwrap().dense_data().unwrap().to_vec())
            .collect::<Vec<_>>()
    )
}

mod host_allocations {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    thread_local! {
        static ACTIVE: Cell<bool> = const { Cell::new(false) };
        static CALLS: Cell<usize> = const { Cell::new(0) };
        static BYTES: Cell<usize> = const { Cell::new(0) };
    }
    pub(super) struct Counting;
    fn record(bytes: usize) {
        let _ = ACTIVE.try_with(|active| {
            if active.get() {
                let _ = CALLS.try_with(|calls| calls.set(calls.get() + 1));
                let _ = BYTES.try_with(|total| total.set(total.get() + bytes));
            }
        });
    }
    // SAFETY: all operations delegate unchanged to System; thread-local counting does not allocate.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            record(layout.size());
            unsafe { System.alloc(layout) }
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            record(layout.size());
            unsafe { System.alloc_zeroed(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            record(size);
            unsafe { System.realloc(ptr, layout, size) }
        }
    }
    #[global_allocator]
    static ALLOCATOR: Counting = Counting;
    pub(super) fn begin() {
        CALLS.with(|calls| calls.set(0));
        BYTES.with(|bytes| bytes.set(0));
        ACTIVE.with(|active| active.set(true));
    }
    pub(super) fn end() -> (usize, usize) {
        ACTIVE.with(|active| active.set(false));
        (CALLS.with(Cell::get), BYTES.with(Cell::get))
    }
}

#[derive(Clone, Copy)]
struct Observation {
    elapsed: Duration,
    host_calls: usize,
    host_bytes: usize,
    cuda: CudaTransferStats,
}

impl std::fmt::Display for Observation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "host_elapsed={:?}/{} alloc/{} bytes/{:?}",
            self.elapsed, self.host_calls, self.host_bytes, self.cuda
        )
    }
}

fn observe<T>(run: impl FnOnce() -> T) -> (T, Observation) {
    host_allocations::begin();
    let before = cuda_transfer_stats();
    let start = Instant::now();
    let value = run();
    let elapsed = start.elapsed();
    let after = cuda_transfer_stats();
    let (host_calls, host_bytes) = host_allocations::end();
    let cuda = CudaTransferStats {
        h2d_calls: after.h2d_calls - before.h2d_calls,
        h2d_bytes: after.h2d_bytes - before.h2d_bytes,
        d2h_calls: after.d2h_calls - before.d2h_calls,
        d2h_bytes: after.d2h_bytes - before.d2h_bytes,
        device_allocs: after.device_allocs - before.device_allocs,
        gemm_calls: after.gemm_calls - before.gemm_calls,
        solver_calls: after.solver_calls - before.solver_calls,
        copy_calls: after.copy_calls - before.copy_calls,
        gauge_ops: after.gauge_ops - before.gauge_ops,
    };
    (
        value,
        Observation {
            elapsed,
            host_calls,
            host_bytes,
            cuda,
        },
    )
}

fn median(mut samples: Vec<Observation>) -> Observation {
    samples.sort_by_key(|sample| sample.elapsed);
    samples[samples.len() / 2]
}

fn check<R: DeviceRule, D: DevicePayload>(case: Case<R, D>) {
    let first_lhs = StackedTensorMap::pack(&[&case.lhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let first_rhs = StackedTensorMap::pack(&[&case.rhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let plan = ContractPlan::new(&first_lhs, &first_rhs, &case.spec()).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let mut second = plan.workspace().unwrap();
    let mut warm_gemms = None;
    for count in [1, 2, 17] {
        let members: Vec<_> = (0..count)
            .map(|i| {
                let lhs = case
                    .lhs
                    .scale(D::entry(1.0 + i as f64 / 8.0, i as f64 / 16.0));
                let rhs = case.rhs.scale(D::entry(1.0 - i as f64 / 32.0, 0.0));
                (lhs, rhs)
            })
            .collect();
        let left: Vec<_> = members.iter().map(|(lhs, _)| lhs).collect();
        let right: Vec<_> = members.iter().map(|(_, rhs)| rhs).collect();
        let lhs = StackedTensorMap::pack(&left).unwrap().to_cuda().unwrap();
        let rhs = StackedTensorMap::pack(&right).unwrap().to_cuda().unwrap();
        let cold_before = cuda_transfer_stats();
        let device_output = plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        let cold_after = cuda_transfer_stats();
        let cold_h2d = cold_after.h2d_calls - cold_before.h2d_calls;
        let cold_bytes = cold_after.h2d_bytes - cold_before.h2d_bytes;
        let cold_gemms = cold_after.gemm_calls - cold_before.gemm_calls;
        assert!(cold_h2d >= 1, "new B needs #740 output zero upload");
        assert!(cold_gemms > 0, "direct core must submit work");
        let output = device_output.to_host().unwrap();
        assert!(
            cold_bytes
                >= (output.member(0).unwrap().dense_data().unwrap().len()
                    * count
                    * std::mem::size_of::<D>()) as u64
        );
        let warm_before = cuda_transfer_stats();
        plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        let warm_after = cuda_transfer_stats();
        assert_eq!(warm_after.h2d_calls - warm_before.h2d_calls, 0);
        assert_eq!(warm_after.d2h_calls - warm_before.d2h_calls, 0);
        let submitted = warm_after.gemm_calls - warm_before.gemm_calls;
        assert!(submitted >= cold_gemms);
        if let Some(expected) = warm_gemms {
            assert_eq!(submitted, expected, "one submission count for every B");
        } else {
            warm_gemms = Some(submitted);
        }
        for (i, (left, right)) in members.iter().enumerate() {
            let member_case = Case {
                name: case.name,
                lhs: left.clone(),
                rhs: right.clone(),
                lhs_axes: case.lhs_axes.clone(),
                rhs_axes: case.rhs_axes.clone(),
                output_axes: case.output_axes.clone(),
                dense: case.dense,
            };
            let oracle = blas_contract_oracle(&member_case);
            let eager = left
                .to_cuda()
                .unwrap()
                .contract(&right.to_cuda().unwrap(), &case.spec())
                .unwrap()
                .to_host()
                .unwrap();
            let actual = output.member(i).unwrap();
            assert_eq!(actual.codomain_rank(), eager.codomain_rank());
            assert_close(
                actual.dense_data().unwrap(),
                oracle.dense_data().unwrap(),
                case.terms(),
                case.name,
            );
            assert_close(
                actual.dense_data().unwrap(),
                eager.dense_data().unwrap(),
                case.terms(),
                case.name,
            );
        }
        let poison: Vec<_> = (0..count).map(|_| poisoned_destination(&case)).collect();
        let mut dst = StackedTensorMap::pack(&poison).unwrap().to_cuda().unwrap();
        let before = cuda_transfer_stats();
        plan.execute_into(&lhs, &rhs, &mut dst, &mut second)
            .unwrap();
        let after = cuda_transfer_stats();
        assert_eq!(after.h2d_calls - before.h2d_calls, 0);
        assert_eq!(after.d2h_calls - before.d2h_calls, 0);
        assert_eq!(after.gemm_calls - before.gemm_calls, submitted);
        assert_eq!(after.copy_calls - before.copy_calls, 0);
        let result = dst.to_host().unwrap();
        for i in 0..count {
            assert_close(
                result.member(i).unwrap().dense_data().unwrap(),
                output.member(i).unwrap().dense_data().unwrap(),
                case.terms(),
                case.name,
            );
        }
        let wrong_indices: &[usize] = if count == 1 { &[0, 0] } else { &[0] };
        let mut wrong_dst = dst.select(wrong_indices).unwrap();
        assert_ne!(wrong_dst.len(), count);
        let before = payload_snapshot(&wrong_dst);
        assert!(matches!(
            plan.execute_into(&lhs, &rhs, &mut wrong_dst, &mut second),
            Err(Error::InvalidArgument(_))
        ));
        assert_eq!(payload_snapshot(&wrong_dst), before);
        assert!(workspace.retained_bytes() > 0);
    }
    drop(plan);
    drop(workspace);
    drop(second);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn public_core_and_swapped_core_match_independent_oracle() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    for (case, _) in candidate_core_probes::<_, f64>(&runtime, &u1_non_self_dual()) {
        if matches!(case.name, "C1" | "C2") {
            check(case);
        }
    }
    for (case, _) in candidate_core_probes::<_, Complex64>(&runtime, &u1_non_self_dual()) {
        if matches!(case.name, "C1" | "C2") {
            check(case);
        }
    }
    for (case, _) in candidate_core_probes::<_, f64>(&runtime, &su2()) {
        if matches!(case.name, "C1" | "C2") {
            check(case);
        }
    }
    for (case, _) in candidate_core_probes::<_, Complex64>(&runtime, &su2()) {
        if matches!(case.name, "C1" | "C2") {
            check(case);
        }
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn swapped_core_uses_rhs_stride_and_nondefault_split() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
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
    let left = StackedTensorMap::pack(&[&lhs, &lhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let right = StackedTensorMap::pack(&[&rhs, &rhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let plan = ContractPlan::new(&left, &right, &spec).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let actual = plan
        .execute(&left, &right, &mut workspace)
        .unwrap()
        .to_host()
        .unwrap();
    let oracle = rhs.compose(&lhs).unwrap();
    for i in 0..2 {
        let member = actual.member(i).unwrap();
        assert_eq!(member.codomain_rank(), 2);
        assert_close(
            member.dense_data().unwrap(),
            oracle.dense_data().unwrap(),
            64,
            "swapped stride",
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn inactive_core_zeroes_each_poisoned_member_with_one_extra_submission() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let case = u1_inactive_cases::<f64>(&runtime)
        .into_iter()
        .next()
        .unwrap();
    let first_lhs = StackedTensorMap::pack(&[&case.lhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let first_rhs = StackedTensorMap::pack(&[&case.rhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let plan = ContractPlan::new(&first_lhs, &first_rhs, &case.spec()).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let oracle = blas_contract_oracle(&case);
    for count in [1, 2, 17, 1] {
        let left = StackedTensorMap::pack(&vec![&case.lhs; count])
            .unwrap()
            .to_cuda()
            .unwrap();
        let right = StackedTensorMap::pack(&vec![&case.rhs; count])
            .unwrap()
            .to_cuda()
            .unwrap();
        let poison = poisoned_destination(&case);
        let mut dst = StackedTensorMap::pack(&vec![&poison; count])
            .unwrap()
            .to_cuda()
            .unwrap();
        let before = cuda_transfer_stats();
        plan.execute(&left, &right, &mut workspace).unwrap();
        let after = cuda_transfer_stats();
        let core_calls = after.gemm_calls - before.gemm_calls;
        assert!(core_calls > 0);
        let before = cuda_transfer_stats();
        plan.execute(&left, &right, &mut workspace).unwrap();
        let after = cuda_transfer_stats();
        assert_eq!(after.gemm_calls - before.gemm_calls, core_calls + 1);
        assert_eq!(after.h2d_calls - before.h2d_calls, 0);
        assert_eq!(after.d2h_calls - before.d2h_calls, 0);
        let before = cuda_transfer_stats();
        plan.execute_into(&left, &right, &mut dst, &mut workspace)
            .unwrap();
        let after = cuda_transfer_stats();
        assert_eq!(
            after.gemm_calls - before.gemm_calls,
            core_calls + 1,
            "one inactive region is zeroed once over all B members"
        );
        assert_eq!(after.h2d_calls - before.h2d_calls, 0);
        assert_eq!(after.d2h_calls - before.d2h_calls, 0);
        let actual = dst.to_host().unwrap();
        for i in 0..count {
            assert_close(
                actual.member(i).unwrap().dense_data().unwrap(),
                oracle.dense_data().unwrap(),
                case.terms(),
                case.name,
            );
        }
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn invalid_bindings_preserve_poisoned_destination() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let other_runtime = Runtime::builder().cuda(0).build().unwrap();
    let reserved_before = runtime
        .cuda_plan_cache_stats()
        .unwrap()
        .unwrap()
        .reserved_entries;
    let case = candidate_core_probes::<_, f64>(&runtime, &u1_non_self_dual())
        .into_iter()
        .find(|(case, _)| case.name == "C1")
        .unwrap()
        .0;
    let lhs = StackedTensorMap::pack(&[&case.lhs, &case.lhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let rhs = StackedTensorMap::pack(&[&case.rhs, &case.rhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let plan = ContractPlan::new(&lhs, &rhs, &case.spec()).unwrap();
    let other = ContractPlan::new(&lhs, &rhs, &case.spec()).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let mut foreign = other.workspace().unwrap();
    let poison = poisoned_destination(&case);
    let mut dst = StackedTensorMap::pack(&[&poison, &poison])
        .unwrap()
        .to_cuda()
        .unwrap();
    let before = payload_snapshot(&dst);
    let short = lhs.select(&[0]).unwrap();
    assert!(plan
        .execute_into(&short, &rhs, &mut dst, &mut workspace)
        .is_err());
    let other_case = candidate_core_probes::<_, f64>(&other_runtime, &u1_non_self_dual())
        .into_iter()
        .find(|(case, _)| case.name == "C1")
        .unwrap()
        .0;
    let wrong_runtime = StackedTensorMap::pack(&[&other_case.lhs, &other_case.lhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    assert!(plan
        .execute_into(&wrong_runtime, &rhs, &mut dst, &mut workspace)
        .is_err());
    let other_space = u1(&[(0, 2)]);
    let wrong_tensor = TensorMap::<_, f64>::zeros(
        &runtime,
        [&other_space, &other_space],
        [&other_space, &other_space],
    )
    .unwrap();
    let wrong_signature = StackedTensorMap::pack(&[&wrong_tensor, &wrong_tensor])
        .unwrap()
        .to_cuda()
        .unwrap();
    assert!(plan
        .execute_into(&wrong_signature, &rhs, &mut dst, &mut workspace)
        .is_err());
    assert!(plan
        .execute_into(&lhs, &rhs, &mut dst, &mut foreign)
        .is_err());
    assert_eq!(payload_snapshot(&dst), before);
    drop(plan);
    drop(other);
    drop(workspace);
    drop(foreign);
    assert_eq!(
        runtime
            .cuda_plan_cache_stats()
            .unwrap()
            .unwrap()
            .reserved_entries,
        reserved_before,
        "workspaces return plan-ledger claims after the plan is dropped"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn negative_only_signed_core_matches_literal_twist() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let odd = tenet::typed::GradedSpace::try_new(
        Arc::new(FermionParityFusionRule.product(U1FusionRule)),
        [(product_sector(Z2Irrep::ODD, U1Irrep::new(0)), 2)],
    )
    .unwrap();
    let odd_dual = odd.try_dual().unwrap();
    let lhs =
        TensorMap::<_, f64>::from_subblock_fn(&runtime, [&odd], [&odd_dual], |_, _| 1.0).unwrap();
    let rhs =
        TensorMap::<_, f64>::from_subblock_fn(&runtime, [&odd_dual], [&odd], |_, _| 1.0).unwrap();
    let case = Case {
        name: "negative-only signed core",
        lhs,
        rhs,
        lhs_axes: vec![1],
        rhs_axes: vec![0],
        output_axes: vec![0, 1],
        dense: false,
    };
    let twist = |tensor: &TensorMap<_, f64>, legs: &[usize]| {
        tensor.twist(legs, Direction::Forward).unwrap()
    };
    let oracle = fermionic_blas_contract_oracle_partitioned(&case, TwistRole::B, 1, twist);
    let untwisted = fermionic_blas_contract_oracle_partitioned(&case, TwistRole::None, 1, twist);
    let expected = oracle.dense_data().unwrap();
    assert!(expected
        .iter()
        .zip(untwisted.dense_data().unwrap())
        .any(|(&x, &y)| (x - y).abs() > 1e-10));
    assert_close(
        case.host().dense_data().unwrap(),
        expected,
        case.terms(),
        case.name,
    );
    let left = StackedTensorMap::pack(&[&case.lhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let right = StackedTensorMap::pack(&[&case.rhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let plan = ContractPlan::new(&left, &right, &case.spec()).unwrap();
    let actual = plan
        .execute(&left, &right, &mut plan.workspace().unwrap())
        .unwrap()
        .to_host()
        .unwrap();
    assert_close(
        actual.member(0).unwrap().dense_data().unwrap(),
        expected,
        case.terms(),
        case.name,
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn copy_c_and_dynamic_routes_are_explicitly_unsupported() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();

    let v = u1(&[(0, 2), (1, 2)]);
    let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&v, &v], 3).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&v, &v], 4).unwrap();
    let left = StackedTensorMap::pack(&[&a]).unwrap().to_cuda().unwrap();
    let right = StackedTensorMap::pack(&[&b]).unwrap().to_cuda().unwrap();
    let spec = ContractSpec {
        lhs: &[2, 3],
        rhs: &[0, 1],
        codomain: &[1, 0],
        domain: &[2, 3],
    };
    assert!(matches!(ContractPlan::new(&left, &right, &spec),
        Err(Error::Operation(error)) if matches!(*error, tenet::typed::OperationError::UnsupportedTensorContractScope { .. })));

    // The second inactive fixture has a source transform and is pinned as
    // DynamicTree by storage_contract_tests::each_way_a_device_overwrite.
    let dynamic = u1_inactive_cases::<f64>(&runtime)
        .into_iter()
        .nth(1)
        .unwrap();
    let left = StackedTensorMap::pack(&[&dynamic.lhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let right = StackedTensorMap::pack(&[&dynamic.rhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    assert!(matches!(ContractPlan::new(&left, &right, &dynamic.spec()),
        Err(Error::Operation(error)) if matches!(*error, tenet::typed::OperationError::UnsupportedTensorContractScope { .. })));
}

#[test]
#[ignore = "requires a real CUDA device"]
fn nonsymmetric_braiding_is_rejected_before_cuda_execution() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let leg =
        tenet::typed::GradedSpace::try_new(Arc::new(RealBraidingProbe::<true>), [(ProbeSector, 2)])
            .unwrap();
    let tensor = TensorMap::<_, f64>::zeros(&runtime, [&leg], [&leg]).unwrap();
    let stack = StackedTensorMap::pack(&[tensor])
        .unwrap()
        .to_cuda()
        .unwrap();
    let spec = ContractSpec {
        lhs: &[1],
        rhs: &[0],
        codomain: &[0],
        domain: &[1],
    };
    assert!(matches!(ContractPlan::new(&stack, &stack, &spec),
    Err(Error::Operation(error)) if matches!(*error,
        tenet::typed::OperationError::UnsupportedTensorContractScope {
            message: tenet::typed::NON_SYMMETRIC_CONTRACTION_UNSUPPORTED
        })));
}

/// Release-only phase observations. Run with `--release --ignored --nocapture
/// --test-threads=1`; timings are evidence, never a CI threshold. Host
/// elapsed times bracket host calls without an explicit device synchronization;
/// they are not end-to-end GPU kernel times. Host allocation counts cover the
/// calling thread and CUDA counters cover TeNeT's submission boundary, not
/// provider-internal kernels or allocations. Fresh output is already zeroed,
/// so warm minus cold GEMM submissions counts inactive-region zero writes.
#[test]
#[ignore = "release-only A100 measurement"]
fn direct_core_release_measurement() {
    let runtime = Runtime::builder()
        .cuda(0)
        .dense_threads(1)
        .gemm_backend(tenet::typed::LinalgBackend::Faer)
        .linalg_backend(tenet::typed::LinalgBackend::Faer)
        .build()
        .unwrap();
    for (case, _) in candidate_core_probes::<_, f64>(&runtime, &su2()) {
        if !matches!(case.name, "C1" | "C2") {
            continue;
        }
        for count in [1, 2, 17] {
            let left: Vec<_> = (0..count)
                .map(|i| case.lhs.scale(1.0 + i as f64 / 8.0))
                .collect();
            let right: Vec<_> = (0..count)
                .map(|i| case.rhs.scale(1.0 - i as f64 / 32.0))
                .collect();
            let ((host_lhs, host_rhs), pack) = observe(|| {
                (
                    StackedTensorMap::pack(&left).unwrap(),
                    StackedTensorMap::pack(&right).unwrap(),
                )
            });
            let ((lhs, rhs), upload) =
                observe(|| (host_lhs.to_cuda().unwrap(), host_rhs.to_cuda().unwrap()));
            let (plan, compile) = observe(|| ContractPlan::new(&lhs, &rhs, &case.spec()).unwrap());
            let cache_before = runtime.cuda_plan_cache_stats().unwrap().unwrap();
            let (mut workspace, reserve) = observe(|| plan.workspace().unwrap());
            let cache_held = runtime.cuda_plan_cache_stats().unwrap().unwrap();
            let (_, cold_returned) = observe(|| {
                black_box(plan.execute(&lhs, &rhs, &mut workspace).unwrap());
            });
            let warm_returned = median(
                (0..11)
                    .map(|_| {
                        observe(|| {
                            black_box(plan.execute(&lhs, &rhs, &mut workspace).unwrap());
                        })
                        .1
                    })
                    .collect(),
            );
            let dest = case.host();
            let mut dst = StackedTensorMap::pack(&vec![&dest; count])
                .unwrap()
                .to_cuda()
                .unwrap();
            let (_, first_into) = observe(|| {
                plan.execute_into(&lhs, &rhs, &mut dst, &mut workspace)
                    .unwrap();
                black_box(&dst);
            });
            let warm_into = median(
                (0..11)
                    .map(|_| {
                        observe(|| {
                            plan.execute_into(&lhs, &rhs, &mut dst, &mut workspace)
                                .unwrap();
                            black_box(&dst);
                        })
                        .1
                    })
                    .collect(),
            );
            let eager_inputs: Vec<_> = left
                .iter()
                .zip(&right)
                .map(|(a, b)| (a.to_cuda().unwrap(), b.to_cuda().unwrap()))
                .collect();
            let eager = median(
                (0..11)
                    .map(|_| {
                        observe(|| {
                            for (a, b) in &eager_inputs {
                                black_box(a.contract(b, &case.spec()).unwrap());
                            }
                        })
                        .1
                    })
                    .collect(),
            );
            let retained = workspace.retained_bytes();
            let scalar_template_bytes = runtime
                .cuda_tree_transform_stats()
                .unwrap()
                .context_scalar_operand_bytes;
            let (_, teardown) = observe(|| {
                drop(workspace);
                drop(plan);
            });
            let cache_after = runtime.cuda_plan_cache_stats().unwrap().unwrap();
            let zero_submissions = warm_returned.cuda.gemm_calls - cold_returned.cuda.gemm_calls;
            eprintln!("{} B={count} f64 CUDA0 Faer1: pack={pack}; upload={upload}; compile={compile}; workspace={reserve}; cold_returned={cold_returned}; warm_returned={warm_returned}; first_into={first_into}; warm_into={warm_into}; eager_members={eager}; zero_region_submissions_per_warm={zero_submissions}; retained_workspace_bytes={retained}; runtime_scalar_template_bytes={scalar_template_bytes}; plan_ledger={}/{}/{}; plan_cache_bytes={}/{}/{}; teardown={teardown}", case.name,
                cache_before.reserved_entries, cache_held.reserved_entries, cache_after.reserved_entries,
                cache_before.retained_bytes, cache_held.retained_bytes, cache_after.retained_bytes);
        }
    }
}

#[test]
#[ignore = "release-only A100 measurement"]
fn signed_core_release_measurement() {
    let runtime = Runtime::builder()
        .cuda(0)
        .dense_threads(1)
        .gemm_backend(tenet::typed::LinalgBackend::Faer)
        .linalg_backend(tenet::typed::LinalgBackend::Faer)
        .build()
        .unwrap();
    let space = fermion_u1();
    let dual = space.try_dual().unwrap();
    let a = TensorMap::<_, f64>::from_subblock_fn(&runtime, [&space], [&dual], fill(201)).unwrap();
    let b = TensorMap::<_, f64>::from_subblock_fn(&runtime, [&dual], [&space], fill(202)).unwrap();
    for swapped in [false, true] {
        let case = Case {
            name: if swapped {
                "signed swapped"
            } else {
                "signed direct"
            },
            lhs: if swapped { b.clone() } else { a.clone() },
            rhs: if swapped { a.clone() } else { b.clone() },
            lhs_axes: vec![usize::from(!swapped)],
            rhs_axes: vec![usize::from(swapped)],
            output_axes: if swapped { vec![1, 0] } else { vec![0, 1] },
            dense: false,
        };
        for count in [1, 2, 17] {
            let left: Vec<_> = (0..count)
                .map(|i| case.lhs.scale(1.0 + i as f64 / 8.0))
                .collect();
            let right: Vec<_> = (0..count)
                .map(|i| case.rhs.scale(1.0 - i as f64 / 32.0))
                .collect();
            let ((host_lhs, host_rhs), pack) = observe(|| {
                (
                    StackedTensorMap::pack(&left).unwrap(),
                    StackedTensorMap::pack(&right).unwrap(),
                )
            });
            let ((lhs, rhs), upload) =
                observe(|| (host_lhs.to_cuda().unwrap(), host_rhs.to_cuda().unwrap()));
            let (plan, compile) = observe(|| ContractPlan::new(&lhs, &rhs, &case.spec()).unwrap());
            let ledger_before = runtime.cuda_plan_cache_stats().unwrap().unwrap();
            let (mut workspace, reserve) = observe(|| plan.workspace().unwrap());
            let ledger_held = runtime.cuda_plan_cache_stats().unwrap().unwrap();
            let (_, cold) = observe(|| {
                black_box(plan.execute(&lhs, &rhs, &mut workspace).unwrap());
            });
            let warm = median(
                (0..11)
                    .map(|_| {
                        observe(|| {
                            black_box(plan.execute(&lhs, &rhs, &mut workspace).unwrap());
                        })
                        .1
                    })
                    .collect(),
            );
            let dest = case.host();
            let mut dst = StackedTensorMap::pack(&vec![&dest; count])
                .unwrap()
                .to_cuda()
                .unwrap();
            let (_, first_into) = observe(|| {
                plan.execute_into(&lhs, &rhs, &mut dst, &mut workspace)
                    .unwrap()
            });
            let warm_into = median(
                (0..11)
                    .map(|_| {
                        observe(|| {
                            plan.execute_into(&lhs, &rhs, &mut dst, &mut workspace)
                                .unwrap();
                        })
                        .1
                    })
                    .collect(),
            );
            let eager_inputs: Vec<_> = left
                .iter()
                .zip(&right)
                .map(|(a, b)| (a.to_cuda().unwrap(), b.to_cuda().unwrap()))
                .collect();
            let eager = median(
                (0..11)
                    .map(|_| {
                        observe(|| {
                            for (a, b) in &eager_inputs {
                                black_box(a.contract(b, &case.spec()).unwrap());
                            }
                        })
                        .1
                    })
                    .collect(),
            );
            let retained = workspace.retained_bytes();
            let (_, teardown) = observe(|| {
                drop(workspace);
                drop(plan);
            });
            let ledger_after = runtime.cuda_plan_cache_stats().unwrap().unwrap();
            eprintln!("{} B={count} f64 CUDA0 Faer1 unsynchronized_host_submission: pack={pack}; upload={upload}; compile={compile}; workspace={reserve}; cold={cold}; warm_median={warm}; first_into={first_into}; warm_into_median={warm_into}; eager_members={eager}; retained_workspace_bytes={retained}; plan_ledger={}/{}/{}; plan_cache_bytes={}/{}/{}; teardown={teardown}",
                case.name, ledger_before.reserved_entries, ledger_held.reserved_entries, ledger_after.reserved_entries,
                ledger_before.retained_bytes, ledger_held.retained_bytes, ledger_after.retained_bytes);
        }
    }
}
