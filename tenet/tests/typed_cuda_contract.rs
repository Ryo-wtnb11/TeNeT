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
//!    Host by the ungated `typed_contract_host_oracle.rs`;
//! 4. fermionic providers (G2c-2, #1347): device == Host == TensorKit's
//!    sequence with the twist on either role, and the TensorKit-valued FZ2
//!    loops as explicit device `contract` calls;
//! 5. `contract_overwrite_into` (G2c-1b, #1346): into a NaN-poisoned device
//!    destination, device == Host eager `contract` (whose overwrite twin is
//!    pinned ungated) == the same independent oracles, on every fixture
//!    above plus destinations with blocks no GEMM writes; warm calls transfer
//!    and allocate nothing; rejections leave the destination untouched.
//!
//! The contract tests read the process-wide transfer counters, hence
//! `--test-threads=1`. Run with `cargo test -p tenet-rs --no-default-features
//! --features cuda,cpu-faer --test typed_cuda_contract -- --ignored
//! --test-threads=1` on a CUDA host.

#![cfg(feature = "cuda")]

mod common;
#[macro_use]
mod contract_cases;

use common::{DevicePayload, DeviceRule};
use contract_cases::{
    assert_close, blas_contract_oracle, candidate_core_probes, dense_oracle, fermion_su2,
    fermion_u1, fermionic_blas_contract_oracle, fermionic_general, fz2_tensorkit_loops, lazy_cases,
    poisoned_destination, product_general, su2, su2_bent, su2_reordered, su2_structure_cases,
    u1_inactive_cases, u1_lhs_identity, u1_non_self_dual, u1_rank_five, u1_reordered,
    u1_rhs_identity, Case, FermionU1, TwistRole,
};
use num_complex::{Complex32, Complex64};
use tenet::dense::{cuda_transfer_stats, CudaTransferStats};
use tenet::typed::{Runtime, TensorMap};

/// Counts Host allocations made by the thread that set `COUNTING` (device
/// runtime threads never do), for the warm host-allocation contract.
mod host_allocations {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static COUNTING: Cell<bool> = const { Cell::new(false) };
        static CALLS: Cell<u64> = const { Cell::new(0) };
    }

    struct Counting;

    fn record() {
        // `try_with`: the allocator also runs during thread teardown.
        let _ = COUNTING.try_with(|counting| {
            if counting.get() {
                let _ = CALLS.try_with(|calls| calls.set(calls.get() + 1));
            }
        });
    }

    // SAFETY: every call forwards unchanged to `System`; the bookkeeping only
    // touches thread-local `Cell`s and never allocates.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            record();
            unsafe { System.alloc(layout) }
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            record();
            unsafe { System.alloc_zeroed(layout) }
        }
        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            unsafe { System.dealloc(pointer, layout) }
        }
        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            record();
            unsafe { System.realloc(pointer, layout, size) }
        }
    }

    #[global_allocator]
    static ALLOCATOR: Counting = Counting;

    /// Host allocation calls `body` makes on this thread.
    pub fn count(body: impl FnOnce()) -> u64 {
        CALLS.with(|calls| calls.set(0));
        COUNTING.with(|counting| counting.set(true));
        body();
        COUNTING.with(|counting| counting.set(false));
        CALLS.with(Cell::get)
    }
}

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

/// The zero-copy sorted/swapped candidates (#1468) run the device Core
/// route, swapped operands and lazy-adjoint GEMM flags included. The
/// fermionic probes take the twisted `blas_contract!` oracle: C2 and L5
/// contract dual B legs in their literal sequence, which TensorKit twists.
#[test]
#[ignore = "requires a real CUDA device"]
fn zero_copy_candidates_match_the_host_at_every_dtype() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    fn at<D: DevicePayload>(runtime: &Runtime) {
        for (case, _) in candidate_core_probes::<_, D>(runtime, &u1_non_self_dual()) {
            check(case);
        }
        for (case, _) in candidate_core_probes::<_, D>(runtime, &su2()) {
            check(case);
        }
        let twist = |t: &TensorMap<FermionU1, D>, legs: &[usize]| t.twist(legs).unwrap();
        for (case, _) in candidate_core_probes::<_, D>(runtime, &fermion_u1()) {
            check_fermionic(case, twist);
        }
    }
    at::<f64>(&runtime);
    at::<Complex64>(&runtime);
    at::<f32>(&runtime);
    at::<Complex32>(&runtime);
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

fn device_contract<R: DeviceRule, D: DevicePayload>(case: &Case<R, D>) -> TensorMap<R, D> {
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
}

/// Device == Host and device == TensorKit's `blas_contract!` with the twist
/// on the B role and on the A role (G2c-2, #1347).
fn check_fermionic<R, D>(
    case: Case<R, D>,
    twist: impl Fn(&TensorMap<R, D>, &[usize]) -> TensorMap<R, D> + Copy,
) where
    R: DeviceRule,
    D: DevicePayload,
{
    let device = device_contract(&case);
    let host = case.host();
    assert_eq!(
        device.codomain_rank(),
        host.codomain_rank(),
        "{}",
        case.name
    );
    assert_close(device.data(), host.data(), case.terms(), case.name);
    for role in [TwistRole::B, TwistRole::A] {
        let oracle = fermionic_blas_contract_oracle(&case, role, twist);
        assert_close(
            device.data(),
            oracle.data(),
            case.terms(),
            &format!("{} vs the {role:?}-role oracle", case.name),
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn fermionic_contractions_match_the_host_and_both_tensorkit_twist_roles_at_every_dtype() {
    // General axes with the twist on one or both contracted legs, the
    // canonical form whose θ varies within one coupled sector (lifted from
    // the storage core route to DynamicTree), and lazy adjoints on either
    // side; fZ2 x U(1) and fZ2 (x) SU(2). Non-vacuity — the twist changes
    // every one of these results — is pinned on the Host, ungated.
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    for_each_fermionic_fixture!(&runtime, f64, check_fermionic);
    for_each_fermionic_fixture!(&runtime, Complex64, check_fermionic);
    for_each_fermionic_fixture!(&runtime, f32, check_fermionic);
    for_each_fermionic_fixture!(&runtime, Complex32, check_fermionic);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn fz2_loops_as_explicit_device_contracts_match_tensorkit() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let loops = fz2_tensorkit_loops(
        &runtime,
        |tensor| tensor.to_cuda().unwrap(),
        |x, y, lhs, rhs, output| x.contract(y, lhs, rhs, output).unwrap(),
        |tensor| tensor.to_host().unwrap().scalar().unwrap(),
    );
    for (name, value, expected) in loops {
        assert!(
            (value - expected).abs() < 1e-12,
            "{name}: {value} vs {expected}"
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_warm_fermionic_contraction_uploads_only_its_output() {
    // What: θ travels as descriptor scalars, so a warm twisted contraction
    // costs exactly what an untwisted one does — the #740 output upload.
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let case: Case<_, f64> = fermionic_general(&runtime, &fermion_su2(), true, "fZ2xSU2", 51);
    let output_bytes = std::mem::size_of_val(case.host().data()) as u64;
    let lhs = case.lhs.to_cuda().unwrap();
    let rhs = case.rhs.to_cuda().unwrap();
    let call = || {
        lhs.contract(&rhs, &case.lhs_axes, &case.rhs_axes, &case.output_axes)
            .unwrap()
    };
    let _ = call();
    let scratch = runtime.cuda_contract_scratch_bytes().unwrap();
    let transforms = runtime.cuda_tree_transform_stats().unwrap();
    let (result, warm) = delta(call);
    assert_eq!(warm.h2d_calls, 1, "{warm:?}");
    assert_eq!(warm.h2d_bytes, output_bytes, "{warm:?}");
    assert_eq!(warm.d2h_calls, 0, "{warm:?}");
    assert_eq!(warm.device_allocs, 1, "{warm:?}");
    assert_eq!(runtime.cuda_contract_scratch_bytes().unwrap(), scratch);
    assert_eq!(runtime.cuda_tree_transform_stats().unwrap(), transforms);
    assert_close(
        result.to_host().unwrap().data(),
        case.host().data(),
        case.terms(),
        "warm",
    );
    println!("fZ2xSU2 mixed θ f64: output {output_bytes} B; warm {warm:?}; {transforms:?}");
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

// ---------------------------------------------------------------------------
// `contract_overwrite_into` (G2c-1b, #1346)
// ---------------------------------------------------------------------------

/// Device `contract_overwrite_into` of `case` into a NaN-poisoned device
/// destination, downloaded.
fn device_overwrite<R: DeviceRule, D: DevicePayload>(case: &Case<R, D>) -> TensorMap<R, D> {
    let mut destination = poisoned_destination(case).to_cuda().unwrap();
    case.lhs
        .to_cuda()
        .unwrap()
        .contract_overwrite_into(
            &case.rhs.to_cuda().unwrap(),
            &mut destination,
            &case.lhs_axes,
            &case.rhs_axes,
            &case.output_axes,
            D::entry(1.0, 0.0),
        )
        .unwrap();
    destination.to_host().unwrap()
}

fn check_overwrite<R: DeviceRule, D: DevicePayload>(case: Case<R, D>) {
    let written = device_overwrite(&case);
    assert_close(written.data(), case.host().data(), case.terms(), case.name);
    let oracle = blas_contract_oracle(&case);
    assert_close(written.data(), oracle.data(), case.terms(), case.name);
}

fn check_overwrite_fermionic<R, D>(
    case: Case<R, D>,
    twist: impl Fn(&TensorMap<R, D>, &[usize]) -> TensorMap<R, D> + Copy,
) where
    R: DeviceRule,
    D: DevicePayload,
{
    let written = device_overwrite(&case);
    assert_close(written.data(), case.host().data(), case.terms(), case.name);
    for role in [TwistRole::B, TwistRole::A] {
        let oracle = fermionic_blas_contract_oracle(&case, role, twist);
        assert_close(
            written.data(),
            oracle.data(),
            case.terms(),
            &format!("{} overwrite vs the {role:?}-role oracle", case.name),
        );
    }
}

fn every_overwrite_fixture<D: DevicePayload>(runtime: &Runtime) {
    check_overwrite(u1_rank_five::<D>(runtime));
    check_overwrite(u1_reordered::<D>(runtime));
    check_overwrite(su2_reordered::<D>(runtime));
    check_overwrite(su2_bent::<D>(runtime));
    check_overwrite(product_general::<D>(runtime));
    check_overwrite(u1_lhs_identity::<D>(runtime));
    check_overwrite(u1_rhs_identity::<D>(runtime));
    for case in u1_inactive_cases::<D>(runtime) {
        check_overwrite(case);
    }
    for case in lazy_cases(&u1_rank_five::<D>(runtime).lhs, "U(1) rank 5 lazy") {
        check_overwrite(case);
    }
    for case in lazy_cases(&su2_reordered::<D>(runtime).lhs, "SU(2) lazy") {
        check_overwrite(case);
    }
    for case in lazy_cases(&product_general::<D>(runtime).lhs, "U(1) x SU(2) lazy") {
        check_overwrite(case);
    }
    for case in su2_structure_cases::<D>(runtime) {
        check_overwrite(case);
    }
    for_each_fermionic_fixture!(runtime, D, check_overwrite_fermionic);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn overwrite_into_a_poisoned_destination_matches_the_host_and_the_oracles_at_every_dtype() {
    // What: general axes, lazy adjoints, the Host-`Structure` class, both
    // fermionic twist roles, and destinations with blocks no GEMM writes —
    // by the core route, by an identity output and under an output transform
    // — each rewritten over NaN.
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    every_overwrite_fixture::<f64>(&runtime);
    every_overwrite_fixture::<Complex64>(&runtime);
    every_overwrite_fixture::<f32>(&runtime);
    every_overwrite_fixture::<Complex32>(&runtime);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_warm_overwrite_transfers_and_allocates_nothing() {
    // What: no reset and no output initialisation — a warm overwrite costs
    // no upload, no download, no device allocation, no scratch growth and no
    // transform preparation, for a transformed general contraction, a
    // twisted one, and a core-route destination with an inactive block.
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    fn warm<R: DeviceRule>(runtime: &Runtime, case: Case<R, f64>) {
        let lhs = case.lhs.to_cuda().unwrap();
        let rhs = case.rhs.to_cuda().unwrap();
        let mut destination = poisoned_destination(&case).to_cuda().unwrap();
        let mut call = || {
            lhs.contract_overwrite_into(
                &rhs,
                &mut destination,
                &case.lhs_axes,
                &case.rhs_axes,
                &case.output_axes,
                1.0,
            )
            .unwrap()
        };
        call();
        let scratch = runtime.cuda_contract_scratch_bytes().unwrap();
        let transforms = runtime.cuda_tree_transform_stats().unwrap();
        let plans = runtime.cuda_plan_cache_stats().unwrap().unwrap();
        let ((), counters) = delta(&mut call);
        let after_plans = runtime.cuda_plan_cache_stats().unwrap().unwrap();
        // #1348: a warm overwrite builds no cuTENSOR plan, observed through
        // the typed Runtime accessor rather than the executor.
        assert_eq!(
            (after_plans.misses, after_plans.evictions),
            (plans.misses, plans.evictions),
            "{}: cuTENSOR plan cache",
            case.name
        );
        assert!(after_plans.hits > plans.hits, "{}: vacuous", case.name);
        // #1348: the warm Host allocation count is a steady state — the
        // same on every warm call. The Host route is compiled on every call
        // (#1359 removed its rank-sized allocations); the inactive-region
        // list is rewritten in place in the lease's contract scratch.
        let host_allocations = [
            host_allocations::count(&mut call),
            host_allocations::count(&mut call),
            host_allocations::count(&mut call),
        ];
        eprintln!("{}: warm host allocations {host_allocations:?}", case.name);
        assert!(host_allocations[0] > 0, "{}: vacuous count", case.name);
        assert!(
            host_allocations
                .iter()
                .all(|&count| count == host_allocations[0]),
            "{}: warm host allocations {host_allocations:?} are not steady",
            case.name
        );
        assert_eq!(
            (
                counters.h2d_calls,
                counters.d2h_calls,
                counters.device_allocs
            ),
            (0, 0, 0),
            "{}: {counters:?}",
            case.name
        );
        assert!(counters.gemm_calls > 0, "{}: vacuous", case.name);
        assert_eq!(runtime.cuda_contract_scratch_bytes().unwrap(), scratch);
        assert_eq!(runtime.cuda_tree_transform_stats().unwrap(), transforms);
        assert_close(
            destination.to_host().unwrap().data(),
            case.host().data(),
            case.terms(),
            case.name,
        );
    }
    warm(&runtime, u1_rank_five::<f64>(&runtime));
    warm(
        &runtime,
        fermionic_general(&runtime, &fermion_su2(), true, "fZ2xSU2 mixed θ", 51),
    );
    let [core, identity_output, output_transform] = u1_inactive_cases::<f64>(&runtime);
    warm(&runtime, core);
    warm(&runtime, identity_output);
    warm(&runtime, output_transform);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn overwrite_rejections_match_the_host_in_order_and_leave_the_destination_untouched() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let other = Runtime::builder().cuda(0).build().unwrap();
    let case = u1_rank_five::<f64>(&runtime);
    let (host_lhs, host_rhs) = (&case.lhs, &case.rhs);
    let lhs = host_lhs.to_cuda().unwrap();
    let rhs = host_rhs.to_cuda().unwrap();
    let poison = || case.host().scale(7.5);
    let poison_data = poison().data().to_vec();
    let _ = lhs
        .contract(&rhs, &case.lhs_axes, &case.rhs_axes, &case.output_axes)
        .unwrap();
    let transforms = runtime.cuda_tree_transform_stats().unwrap();
    let scratch = runtime.cuda_contract_scratch_bytes().unwrap();
    let device_text = |text: String| text.replace("host storage", "CUDA storage");

    // Host-shared rejections, each against a fresh poisoned destination:
    // malformed axes, an output order that is not a permutation, a pairing
    // of two equal (not dual) legs, and a destination of another space.
    let malformed: [(&[usize], &[usize], &[usize]); 4] = [
        (&[3, 7], &[0, 3], &[2, 0, 4, 1, 3]),
        (&[3, 3], &[0, 3], &[2, 0, 4, 1, 3]),
        (&[3, 1], &[0, 3], &[2, 0, 4, 1, 1]),
        (&[0], &[0], &[0, 1, 2, 3, 4, 5, 6]),
    ];
    let foreign = || u1_reordered::<f64>(&runtime).host().scale(7.5);
    let cases = malformed.into_iter().map(|axes| (axes, poison())).chain([(
        (
            &case.lhs_axes[..],
            &case.rhs_axes[..],
            &case.output_axes[..],
        ),
        foreign(),
    )]);
    for ((lhs_axes, rhs_axes, output), host_destination) in cases {
        let before = host_destination.data().to_vec();
        let mut destination = host_destination.to_cuda().unwrap();
        let mut host_destination = host_destination;
        let expected = host_lhs
            .contract_overwrite_into(
                host_rhs,
                &mut host_destination,
                lhs_axes,
                rhs_axes,
                output,
                1.0,
            )
            .unwrap_err()
            .to_string();
        let (actual, counters) = delta(|| {
            lhs.contract_overwrite_into(&rhs, &mut destination, lhs_axes, rhs_axes, output, 1.0)
                .unwrap_err()
                .to_string()
        });
        assert_eq!(actual, device_text(expected), "{lhs_axes:?} {output:?}");
        assert_eq!(counters, CudaTransferStats::default(), "{actual}");
        assert_eq!(destination.to_host().unwrap().data(), before.as_slice());
    }

    // Runtime, shared ownership and alias, then the device's own alpha
    // boundary.
    let mut destination = poison().to_cuda().unwrap();
    let stranger = u1_rank_five::<f64>(&other).rhs.to_cuda().unwrap();
    let shared = destination.clone();
    let (errors, counters) = delta(|| {
        let axes = (
            &case.lhs_axes[..],
            &case.rhs_axes[..],
            &case.output_axes[..],
        );
        let mut errors = vec![
            lhs.contract_overwrite_into(&stranger, &mut destination, axes.0, axes.1, axes.2, 1.0)
                .unwrap_err(),
            lhs.contract_overwrite_into(&rhs, &mut destination, axes.0, axes.1, axes.2, 1.0)
                .unwrap_err(),
        ];
        let mut lhs_alias = lhs.clone();
        errors.push(
            lhs.contract_overwrite_into(&rhs, &mut lhs_alias, axes.0, axes.1, axes.2, 1.0)
                .unwrap_err(),
        );
        errors
    });
    assert!(
        matches!(errors[0], tenet::prelude::Error::RuntimeMismatch),
        "{:?}",
        errors[0]
    );
    assert!(
        errors[1].to_string().contains("uniquely owned"),
        "{}",
        errors[1]
    );
    assert!(errors[2].to_string().contains("alias"), "{}", errors[2]);
    drop(shared);
    let (alpha, alpha_counters) = delta(|| {
        lhs.contract_overwrite_into(
            &rhs,
            &mut destination,
            &case.lhs_axes,
            &case.rhs_axes,
            &case.output_axes,
            2.0,
        )
        .unwrap_err()
    });
    assert!(
        matches!(alpha, tenet::prelude::Error::UnsupportedOnDevice(_)),
        "{alpha:?}"
    );
    assert_eq!(counters, CudaTransferStats::default(), "{counters:?}");
    assert_eq!(
        alpha_counters,
        CudaTransferStats::default(),
        "{alpha_counters:?}"
    );
    assert_eq!(
        destination.to_host().unwrap().data(),
        poison_data.as_slice()
    );
    assert_eq!(runtime.cuda_tree_transform_stats().unwrap(), transforms);
    assert_eq!(runtime.cuda_contract_scratch_bytes().unwrap(), scratch);
}
