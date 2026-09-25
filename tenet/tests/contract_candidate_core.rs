//! Zero-copy TensorKit contraction candidates (#1468).
//!
//! TensorKit `contract!` (tensoroperations.jl L318-357) scores the lhs- and
//! rhs-sorted pairings in both operand orders, and
//! `has_shared_permute(::AdjointTensorMap)` (indexmanipulations.jl L282)
//! makes a lazy adjoint free when its permuted order is its natural adjoint
//! order. The audit probes of `reviews/contract-candidate-selection-audit`
//! are free in TensorKit only after that sort or swap:
//!
//! - C1 `A[3,2]·B[1,0]`, C2 `A[0,1]·B[2,3] → [2,3,0,1]` (owned);
//! - L3 `P'[3,2]·B[1,0]`, L5 `A[0,1]·P'[2,3] → [2,3,0,1]`,
//!   L7 `A[3,2]·P'[1,0]` (`P'` a lazy adjoint).
//!
//! Each must agree with TensorKit's `blas_contract!` step sequence on Host
//! typed operations (which materializes through `permute`, never through
//! the contraction route under test), run no tree transform on a warm call,
//! and allocate no more than its literal Core control (C0 `A[2,3]·B[0,1]`,
//! L4 `P'[2,3]·B[0,1]`).

mod common;
#[allow(unused_macros)]
mod contract_cases;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::Mutex;

use contract_cases::{
    assert_close, blas_contract_oracle, candidate_core_probes as probes, fermion_u1,
    fermionic_blas_contract_oracle, fill, su2, u1_non_self_dual, Case, FermionU1, Payload,
    TwistRole,
};
use num_complex::{Complex32, Complex64};
use tenet::core::{CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec};
use tenet::typed::{GradedSpace, Runtime, TensorMap};

struct CountingAllocator;

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<u64> = const { Cell::new(0) };
    static BYTES: Cell<u64> = const { Cell::new(0) };
}

fn record(bytes: usize) {
    if ENABLED.get() {
        CALLS.set(CALLS.get() + 1);
        BYTES.set(BYTES.get() + bytes as u64);
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() {
            record(new_size);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;
// Why serialize: process-global layout caches are shared across test threads,
// so a concurrent test's evictions would move this thread's warm counts.
static MEASUREMENT_LOCK: Mutex<()> = Mutex::new(());

/// Values against `oracle`, then the warm route evidence on `f64`.
fn check<R, D>(
    runtime: &Runtime,
    v: &GradedSpace<R>,
    symmetry: &str,
    oracle: impl Fn(&Case<R, D>) -> TensorMap<R, D>,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    for (case, _) in probes::<R, D>(runtime, v) {
        let expected = oracle(&case);
        assert_close(
            case.host().data(),
            expected.data(),
            case.terms(),
            &format!("{symmetry} {}", case.name),
        );
    }
}

/// Warm `(allocation calls, allocation bytes, tree-transform lookups)`.
fn warm<R>(runtime: &Runtime, case: &Case<R, f64>) -> (u64, u64, usize)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let call = || {
        black_box(&case.lhs)
            .contract(&case.rhs, &case.lhs_axes, &case.rhs_axes, &case.output_axes)
            .unwrap()
    };
    call();
    call();
    let lookups = || {
        let info = runtime.tree_transform_cache_info();
        info.hits() + info.misses()
    };
    let before = lookups();
    CALLS.set(0);
    BYTES.set(0);
    ENABLED.set(true);
    let result = call();
    ENABLED.set(false);
    drop(result);
    (CALLS.get(), BYTES.get(), lookups() - before)
}

fn assert_warm_zero_copy<R>(v: &GradedSpace<R>, symmetry: &str)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let _guard = MEASUREMENT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let probes = probes::<R, f64>(&runtime, v);
    let measured: Vec<_> = probes
        .iter()
        .map(|(case, control)| (case.name, *control, warm(&runtime, case)))
        .collect();
    let control = |name| {
        measured
            .iter()
            .find(|(probe, _, _)| *probe == name)
            .unwrap()
            .2
    };
    for &(name, control_name, (calls, bytes, lookups)) in &measured {
        let (control_calls, control_bytes, _) = control(control_name);
        eprintln!("{symmetry} {name}: {calls} calls, {bytes} B, {lookups} transform lookups");
        // What: the selected candidate transforms neither source nor output.
        assert_eq!(lookups, 0, "{symmetry} {name}: tree transforms ran");
        // What: no more allocation than the literal Core call.
        assert!(
            calls <= control_calls && bytes <= control_bytes,
            "{symmetry} {name}: {calls} calls / {bytes} B vs {control_name} \
             {control_calls} / {control_bytes}"
        );
    }
}

#[test]
fn zero_copy_candidates_run_no_transform_and_allocate_like_literal_core() {
    assert_warm_zero_copy(&u1_non_self_dual(), "U(1)");
    assert_warm_zero_copy(&su2(), "SU(2)");
    assert_warm_zero_copy(&fermion_u1(), "fZ2xU(1)");
}

fn bosonic_values<D: Payload>() {
    let runtime = Runtime::builder().build().unwrap();
    check::<_, D>(&runtime, &u1_non_self_dual(), "U(1)", blas_contract_oracle);
    check::<_, D>(&runtime, &su2(), "SU(2)", blas_contract_oracle);
}

#[test]
fn zero_copy_candidates_match_the_tensorkit_blas_contract_sequence() {
    bosonic_values::<f64>();
    bosonic_values::<Complex64>();
    bosonic_values::<f32>();
    bosonic_values::<Complex32>();
}

fn fermionic_values<D: Payload>() {
    let runtime = Runtime::builder().build().unwrap();
    let twist = |t: &TensorMap<FermionU1, D>, legs: &[usize]| t.twist(legs).unwrap();
    check::<_, D>(&runtime, &fermion_u1(), "fZ2xU(1)", |case| {
        fermionic_blas_contract_oracle(case, TwistRole::B, twist)
    });

    // Twisted control: `A: V⊗V ← V*⊗V*` with `B` (or `P'`) `V*⊗V* ← V⊗V`
    // puts dual legs on the core-right contracted codomain of the sorted
    // candidate. TensorKit copies an operand to twist it, and the result
    // must match TensorKit's B-role twist, not the untwisted sequence.
    let v = fermion_u1();
    let v_dual = v.try_dual().unwrap();
    let a =
        TensorMap::<_, D>::from_block_fn(&runtime, [&v, &v], [&v_dual, &v_dual], fill(4)).unwrap();
    let b =
        TensorMap::<_, D>::from_block_fn(&runtime, [&v_dual, &v_dual], [&v, &v], fill(5)).unwrap();
    let p =
        TensorMap::<_, D>::from_block_fn(&runtime, [&v, &v], [&v_dual, &v_dual], fill(6)).unwrap();
    for (name, rhs) in [("twisted C1", b), ("twisted L7", p.adjoint().unwrap())] {
        let case = Case {
            name,
            lhs: a.clone(),
            rhs,
            lhs_axes: vec![3, 2],
            rhs_axes: vec![1, 0],
            output_axes: vec![0, 1, 2, 3],
            dense: false,
        };
        let host = case.host();
        let twisted = fermionic_blas_contract_oracle(&case, TwistRole::B, twist);
        assert_close(host.data(), twisted.data(), case.terms(), name);
        let untwisted = fermionic_blas_contract_oracle(&case, TwistRole::None, twist);
        assert!(
            host.data()
                .iter()
                .zip(untwisted.data())
                .any(|(&x, &y)| x.distance(y) > 1e-3),
            "{name}: the twist changes nothing here"
        );
    }
}

#[test]
fn zero_copy_fermionic_candidates_match_tensorkit_with_and_without_twist() {
    fermionic_values::<f64>();
    fermionic_values::<Complex64>();
    fermionic_values::<f32>();
    fermionic_values::<Complex32>();
}
