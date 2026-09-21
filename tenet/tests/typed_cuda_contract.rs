//! Real-device gates for general-axes contraction on device tensors
//! (G2c-1a, issue #1345): the Host `DynamicTree` artifact replayed by device
//! executors.
//!
//! Evidence chain:
//!
//! 1. the executor replaying one artifact is gated against the Host replay of
//!    the *same* artifact for both forced orientations and every axis-order
//!    candidate in `tenet-tensors` (`storage_contract_tests.rs`), where the
//!    test-only plan builder can force them; the NaN-poisoned core-destination
//!    scratch is gated there too;
//! 2. the typed lowering is gated here: device == Host of the same public
//!    call, at every device payload dtype, owned and lazy-adjoint;
//! 3. independence: device == TensorKit's `blas_contract!` step sequence on
//!    Host typed ops (bosonic providers), and device == the physical-basis
//!    dense contraction (U(1), SU(2)). Both oracles are pinned against the
//!    Host by the ungated `typed_contract_host_oracle.rs`.
//!
//! The contract tests read the process-wide transfer counters, hence
//! `--test-threads=1`. Run with `cargo test -p tenet-rs --no-default-features
//! --features cuda,cpu-faer --test typed_cuda_contract -- --ignored
//! --test-threads=1` on a CUDA host.

#![cfg(feature = "cuda")]

mod common;
mod contract_cases;

use std::sync::Arc;

use common::{DevicePayload, DeviceRule};
use contract_cases::{
    assert_close, blas_contract_oracle, dense_oracle, fill, lazy_cases, product_general, su2_bent,
    su2_reordered, su2_structure_cases, u1_lhs_identity, u1_rank_five, u1_reordered,
    u1_rhs_identity, Case,
};
use num_complex::{Complex32, Complex64};
use tenet::core::{FermionParityFusionRule, Z2Irrep};
use tenet::dense::{cuda_transfer_stats, CudaTransferStats};
use tenet::typed::{Error, GradedSpace, Runtime, TensorMap};

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

/// Device == Host and device == `blas_contract!` for one fixture.
fn check<R, D>(case: Case<R, D>)
where
    R: DeviceRule,
    D: DevicePayload,
{
    let host = case.host();
    let device = case
        .lhs
        .to_cuda()
        .unwrap()
        .contract(
            &case.rhs.to_cuda().unwrap(),
            &case.lhs_axes,
            &case.rhs_axes,
            &case.output_axes,
        )
        .unwrap()
        .to_host()
        .unwrap();
    assert_eq!(
        device.codomain_rank(),
        host.codomain_rank(),
        "{}",
        case.name
    );
    assert_eq!(device.domain_rank(), host.domain_rank(), "{}", case.name);
    assert_close(device.data(), host.data(), case.terms(), case.name);
    let oracle = blas_contract_oracle(&case);
    assert_close(device.data(), oracle.data(), case.terms(), case.name);
}

fn every_fixture<D: DevicePayload>(runtime: &Runtime) {
    check(u1_rank_five::<D>(runtime));
    check(u1_reordered::<D>(runtime));
    check(su2_reordered::<D>(runtime));
    check(su2_bent::<D>(runtime));
    check(product_general::<D>(runtime));
    check(u1_lhs_identity::<D>(runtime));
    check(u1_rhs_identity::<D>(runtime));
}

#[test]
#[ignore = "requires a real CUDA device"]
fn general_axes_match_the_host_and_the_blas_contract_oracle_at_every_dtype() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    every_fixture::<f64>(&runtime);
    every_fixture::<Complex64>(&runtime);
    every_fixture::<f32>(&runtime);
    every_fixture::<Complex32>(&runtime);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn general_axes_match_the_physical_basis_contraction() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    fn dense<R, D>(case: Case<R, D>)
    where
        R: DeviceRule + tenet::core::PhysicalFusionBasis<Scalar = f64>,
        D: DevicePayload,
    {
        assert!(case.dense, "{}", case.name);
        let (shape, expected) = dense_oracle(&case);
        let device = case
            .lhs
            .to_cuda()
            .unwrap()
            .contract(
                &case.rhs.to_cuda().unwrap(),
                &case.lhs_axes,
                &case.rhs_axes,
                &case.output_axes,
            )
            .unwrap()
            .to_host()
            .unwrap()
            .to_physical_dense()
            .unwrap();
        assert_eq!(device.shape, shape, "{}", case.name);
        assert_close(&device.data, &expected, case.terms(), case.name);
    }
    dense(u1_reordered::<f64>(&runtime));
    dense(u1_reordered::<Complex64>(&runtime));
    dense(su2_reordered::<f64>(&runtime));
    dense(su2_reordered::<Complex64>(&runtime));
}

fn lazy_at<D: DevicePayload>(runtime: &Runtime) {
    for case in lazy_cases(&u1_rank_five::<D>(runtime).lhs, "U(1) rank 5 lazy") {
        check(case);
    }
    for case in lazy_cases(&su2_reordered::<D>(runtime).lhs, "SU(2) lazy") {
        check(case);
    }
    for case in lazy_cases(&product_general::<D>(runtime).lhs, "U(1) x SU(2) lazy") {
        check(case);
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn where_the_host_takes_its_structure_route_the_device_agrees_at_every_dtype() {
    // The device runs the prelowered DynamicTree artifact for this class,
    // a path the Host itself never takes for it (it picks `Structure`).
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    fn at<D: DevicePayload>(runtime: &Runtime) {
        for case in su2_structure_cases::<D>(runtime) {
            check(case);
        }
    }
    at::<f64>(&runtime);
    at::<Complex64>(&runtime);
    at::<f32>(&runtime);
    at::<Complex32>(&runtime);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn lazy_adjoint_operands_match_the_host_at_every_dtype() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    lazy_at::<f64>(&runtime);
    lazy_at::<Complex64>(&runtime);
    lazy_at::<f32>(&runtime);
    lazy_at::<Complex32>(&runtime);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_warm_general_contraction_uploads_only_its_output() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let case = u1_rank_five::<f64>(&runtime);
    let output_bytes = std::mem::size_of_val(case.host().data()) as u64;
    let lhs = case.lhs.to_cuda().unwrap();
    let rhs = case.rhs.to_cuda().unwrap();
    let call = || {
        lhs.contract(&rhs, &case.lhs_axes, &case.rhs_axes, &case.output_axes)
            .unwrap()
    };

    // Cold: the output, the coefficient payloads and the scratch high-water
    // zero uploads.
    let (_, cold) = delta(call);
    assert!(cold.h2d_calls > 1, "{cold:?}");
    assert_eq!(cold.d2h_calls, 0, "{cold:?}");
    let scratch = runtime.cuda_contract_scratch_bytes().unwrap();
    assert!(scratch > 0, "a transformed operand lives in the scratch");
    let transforms = runtime.cuda_tree_transform_stats().unwrap();

    // Warm: exactly the #740 output initialisation.
    let (_, warm) = delta(call);
    assert_eq!(warm.h2d_calls, 1, "{warm:?}");
    assert_eq!(warm.h2d_bytes, output_bytes, "{warm:?}");
    assert_eq!(warm.d2h_calls, 0, "{warm:?}");
    assert_eq!(warm.d2h_bytes, 0, "{warm:?}");
    assert_eq!(warm.device_allocs, 1, "{warm:?}");
    assert_eq!(
        runtime.cuda_contract_scratch_bytes().unwrap(),
        scratch,
        "a warm call grows no scratch"
    );
    assert_eq!(
        runtime.cuda_tree_transform_stats().unwrap(),
        transforms,
        "a warm call prepares no transform state"
    );
    println!(
        "u1_rank_five f64: output {output_bytes} B; cold {cold:?}; warm {warm:?}; \
         scratch {scratch} B; {transforms:?}"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn the_scratch_grows_only_at_a_high_water_mark_and_is_released_by_the_clear_path() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let large = u1_rank_five::<f64>(&runtime);
    let small = u1_reordered::<f64>(&runtime);
    let run = |case: &Case<_, f64>| {
        case.lhs
            .to_cuda()
            .unwrap()
            .contract(
                &case.rhs.to_cuda().unwrap(),
                &case.lhs_axes,
                &case.rhs_axes,
                &case.output_axes,
            )
            .unwrap()
            .to_host()
            .unwrap()
    };
    let _ = run(&large);
    let small_lhs = small.lhs.to_cuda().unwrap();
    let small_rhs = small.rhs.to_cuda().unwrap();
    // Warm the small case's transform structures (and any buffer it needs
    // wider than the large case did) so only its output is left.
    let _ = small_lhs
        .contract(
            &small_rhs,
            &small.lhs_axes,
            &small.rhs_axes,
            &small.output_axes,
        )
        .unwrap();
    let high_water = runtime.cuda_contract_scratch_bytes().unwrap();
    let (result, counters) = delta(|| {
        small_lhs
            .contract(
                &small_rhs,
                &small.lhs_axes,
                &small.rhs_axes,
                &small.output_axes,
            )
            .unwrap()
    });
    // What: a smaller operand narrows the retained buffers instead of
    // reallocating them — one allocation, the returned output.
    assert_eq!(counters.device_allocs, 1, "{counters:?}");
    assert_eq!(counters.h2d_calls, 1, "{counters:?}");
    assert_eq!(runtime.cuda_contract_scratch_bytes().unwrap(), high_water);
    assert_close(
        result.to_host().unwrap().data(),
        small.host().data(),
        small.terms(),
        "narrowed scratch",
    );
    // And the large one again, after the narrowing, is still right.
    assert_close(
        run(&large).data(),
        large.host().data(),
        large.terms(),
        "re-widened",
    );
    assert_eq!(runtime.cuda_contract_scratch_bytes().unwrap(), high_water);

    runtime.clear_tree_transform_cache();
    assert_eq!(runtime.cuda_contract_scratch_bytes().unwrap(), 0);
    assert_close(
        run(&small).data(),
        small.host().data(),
        small.terms(),
        "after clear",
    );
}

fn fermionic_fixture(
    runtime: &Runtime,
) -> (
    TensorMap<FermionParityFusionRule, f64>,
    TensorMap<FermionParityFusionRule, f64>,
) {
    let leg = GradedSpace::try_new_with_arc(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 2)],
    )
    .unwrap();
    let dual = leg.try_dual().unwrap();
    let lhs = TensorMap::from_block_fn(runtime, [&leg, &leg], [&leg], fill(21)).unwrap();
    let rhs = TensorMap::from_block_fn(runtime, [&leg, &dual], [&leg], fill(22)).unwrap();
    (lhs, rhs)
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_fermionic_dual_contracted_leg_is_unsupported_before_any_device_work() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let (lhs, rhs) = fermionic_fixture(&runtime);
    // The Host runs it (its twist is an in-place scale the device lacks).
    let _ = lhs.contract(&rhs, &[2, 0], &[0, 1], &[0, 1]).unwrap();
    let lhs = lhs.to_cuda().unwrap();
    let rhs = rhs.to_cuda().unwrap();
    let transforms = runtime.cuda_tree_transform_stats().unwrap();
    let scratch = runtime.cuda_contract_scratch_bytes().unwrap();
    let (error, counters) = delta(|| lhs.contract(&rhs, &[2, 0], &[0, 1], &[0, 1]).unwrap_err());
    assert!(
        matches!(&error, Error::UnsupportedOnDevice(message) if message.contains("twist")),
        "{error:?}"
    );
    assert_eq!(counters, CudaTransferStats::default(), "{counters:?}");
    assert_eq!(runtime.cuda_tree_transform_stats().unwrap(), transforms);
    assert_eq!(runtime.cuda_contract_scratch_bytes().unwrap(), scratch);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn rejections_happen_before_any_device_work_and_match_the_host() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let other = Runtime::builder().cuda(0).build().unwrap();
    let case = u1_rank_five::<f64>(&runtime);
    let host_lhs = &case.lhs;
    let host_rhs = &case.rhs;
    let lhs = host_lhs.to_cuda().unwrap();
    let rhs = host_rhs.to_cuda().unwrap();
    let stranger = u1_rank_five::<f64>(&other).rhs.to_cuda().unwrap();
    let _ = lhs
        .contract(&rhs, &case.lhs_axes, &case.rhs_axes, &case.output_axes)
        .unwrap();
    let transforms = runtime.cuda_tree_transform_stats().unwrap();
    let scratch = runtime.cuda_contract_scratch_bytes().unwrap();
    // Malformed: an axis out of range, a repeated axis, an output order that
    // is not a permutation, and a pairing of two equal (not dual) legs.
    let (errors, counters) = delta(|| {
        let malformed: [(&[usize], &[usize], &[usize]); 4] = [
            (&[3, 7], &[0, 3], &[2, 0, 4, 1, 3]),
            (&[3, 3], &[0, 3], &[2, 0, 4, 1, 3]),
            (&[3, 1], &[0, 3], &[2, 0, 4, 1, 1]),
            (&[0], &[0], &[0, 1, 2, 3, 4, 5, 6]),
        ];
        let mut errors = Vec::new();
        for (lhs_axes, rhs_axes, output) in malformed {
            let expected = host_lhs
                .contract(host_rhs, lhs_axes, rhs_axes, output)
                .unwrap_err()
                .to_string();
            let actual = lhs
                .contract(&rhs, lhs_axes, rhs_axes, output)
                .unwrap_err()
                .to_string();
            assert_eq!(actual, expected, "device error text must be the Host's");
            errors.push(actual);
        }
        errors.push(
            lhs.contract(&stranger, &case.lhs_axes, &case.rhs_axes, &case.output_axes)
                .unwrap_err()
                .to_string(),
        );
        errors
    });
    assert_eq!(errors.len(), 5);
    assert_eq!(
        counters,
        CudaTransferStats::default(),
        "a rejected contraction must submit nothing: {counters:?}"
    );
    assert_eq!(runtime.cuda_tree_transform_stats().unwrap(), transforms);
    assert_eq!(runtime.cuda_contract_scratch_bytes().unwrap(), scratch);
}
