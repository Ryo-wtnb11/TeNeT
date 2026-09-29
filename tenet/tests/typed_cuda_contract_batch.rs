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
    assert_close, blas_contract_oracle, candidate_core_probes, fill, poisoned_destination, su2, u1,
    u1_inactive_cases, u1_non_self_dual, Case,
};
use num_complex::Complex64;
use std::sync::Arc;
use tenet::expert::cuda_transfer_stats;
use tenet::sector::{
    product_sector, FermionParityFusionRule, ProductFusionRuleExt, U1FusionRule, U1Irrep, Z2Irrep,
};
use tenet::typed::{ContractPlan, ContractSpec, Error, Runtime, StackedTensorMap, TensorMap};

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
        let mut wrong_dst = dst.select(&[0]).unwrap();
        let before = format!(
            "{:?}",
            wrong_dst
                .to_host()
                .unwrap()
                .member(0)
                .unwrap()
                .dense_data()
                .unwrap()
        );
        assert!(matches!(
            plan.execute_into(&lhs, &rhs, &mut wrong_dst, &mut second),
            Err(Error::InvalidArgument(_))
        ));
        assert_eq!(
            format!(
                "{:?}",
                wrong_dst
                    .to_host()
                    .unwrap()
                    .member(0)
                    .unwrap()
                    .dense_data()
                    .unwrap()
            ),
            before
        );
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
    let before = format!(
        "{:?}",
        dst.to_host()
            .unwrap()
            .member(0)
            .unwrap()
            .dense_data()
            .unwrap()
    );
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
    assert_eq!(
        format!(
            "{:?}",
            dst.to_host()
                .unwrap()
                .member(0)
                .unwrap()
                .dense_data()
                .unwrap()
        ),
        before
    );
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
fn signed_copy_c_and_dynamic_routes_are_explicitly_unsupported() {
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
    let left = StackedTensorMap::pack(&[&lhs]).unwrap().to_cuda().unwrap();
    let right = StackedTensorMap::pack(&[&rhs]).unwrap().to_cuda().unwrap();
    let spec = ContractSpec {
        lhs: &[1],
        rhs: &[0],
        codomain: &[0],
        domain: &[1],
    };
    assert!(matches!(ContractPlan::new(&left, &right, &spec),
        Err(Error::Operation(error)) if matches!(*error, tenet::typed::OperationError::UnsupportedTensorContractScope { .. })));

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
