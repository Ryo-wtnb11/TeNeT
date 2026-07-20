use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use num_complex::Complex64;
use tenet::prelude::*;

struct CountingAllocator;

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static BYTES: Cell<u64> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && ENABLED.get() {
            BYTES.set(BYTES.get() + layout.size() as u64);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() && ENABLED.get() {
            BYTES.set(BYTES.get() + new_size as u64);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn measured_bytes<T>(f: impl FnOnce() -> T) -> (T, u64) {
    BYTES.set(0);
    ENABLED.set(true);
    let result = f();
    ENABLED.set(false);
    (result, BYTES.get())
}

fn old_compose_oracle(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let lhs_axes = (lhs.codomain_rank()..lhs.rank()).collect::<Vec<_>>();
    let rhs_axes = (0..rhs.codomain_rank()).collect::<Vec<_>>();
    lhs.contract(&rhs.twist(&rhs_axes).unwrap(), &lhs_axes, &rhs_axes)
        .unwrap()
}

fn assert_c64_close(actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).norm() < 1e-11,
            "element {index}: actual={actual}, expected={expected}"
        );
    }
}

fn assert_f64_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-11,
            "element {index}: actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn fermionic_compose_matches_explicit_twist_cancellation_without_specializing_the_rule() {
    // What: fZ2 and the non-Abelian product both implement TensorKit `mul!`
    // semantics directly, while ordinary tensorcontract retains its twist.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let spaces = [
        Space::fz2([(0, 2), (1, 3)]).unwrap(),
        Space::fz2_u1_su2([((0, 0, 0), 2), ((1, 0, 1), 3)]).unwrap(),
    ];
    for (fixture, space) in spaces.into_iter().enumerate() {
        let lhs = Tensor::rand_with_seed(
            &runtime,
            Dtype::F64,
            [&space],
            [&space.dual()],
            353_100 + fixture as u64 * 10,
        )
        .unwrap();
        let rhs = Tensor::rand_with_seed(
            &runtime,
            Dtype::F64,
            [&space.dual()],
            [&space],
            353_101 + fixture as u64 * 10,
        )
        .unwrap();

        let composed = lhs.compose(&rhs).unwrap();
        let oracle = old_compose_oracle(&lhs, &rhs);
        let tensorcontract = lhs.contract(&rhs, &[1], &[0]).unwrap();

        assert_f64_close(composed.data(), oracle.data());
        assert!(
            composed
                .data()
                .iter()
                .zip(tensorcontract.data())
                .any(|(&a, &b)| (a - b).abs() > 1e-11),
            "fixture {fixture} did not exercise an odd-sector twist"
        );
        assert_eq!(composed.codomain_rank(), 1);
        assert_eq!(composed.domain_rank(), 1);
    }
}

#[test]
fn complex_product_compose_matches_the_old_semantic_sequence_elementwise() {
    // What: raw complex reduced-block values retain their real and imaginary
    // parts when composition omits the cancelling whole-RHS twist.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = Space::fz2_u1_su2([((0, 0, 0), 2), ((1, 0, 1), 2), ((1, 1, 2), 1)]).unwrap();
    let lhs =
        Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space.dual()], 353_201).unwrap();
    let rhs =
        Tensor::rand_with_seed(&runtime, Dtype::C64, [&space.dual()], [&space], 353_202).unwrap();

    let composed = lhs.compose(&rhs).unwrap();
    let oracle = old_compose_oracle(&lhs, &rhs);

    assert!(rhs.data_c64().iter().any(|value| value.im != 0.0));
    assert_c64_close(composed.data_c64(), oracle.data_c64());
}

#[test]
fn lazy_fermionic_rhs_compose_reads_parent_storage_without_materializing_it() {
    // What: a categorical adjoint on the RHS follows the same direct map
    // composition semantics while retaining its shared parent allocation.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = Space::fz2_u1_su2([((0, 0, 0), 2), ((1, 0, 1), 2), ((1, 1, 2), 1)]).unwrap();
    let lhs =
        Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space.dual()], 353_251).unwrap();
    let parent =
        Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space.dual()], 353_252).unwrap();
    let rhs = parent.adjoint().unwrap();
    let shared_before = rhs.storage_strong_count();

    let composed = lhs.compose(&rhs).unwrap();
    let oracle = old_compose_oracle(&lhs, &rhs);

    assert_c64_close(composed.data_c64(), oracle.data_c64());
    assert_eq!(rhs.storage_strong_count(), shared_before);
}

#[test]
fn lazy_lhs_and_both_adjoint_compose_match_raw_complex_oracles() {
    // What: the common Gram path A† * B and the both-adjoint variant map each
    // logical operand independently onto shared parent storage.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = Space::fz2_u1_su2([((0, 0, 0), 2), ((1, 0, 1), 2), ((1, 1, 2), 1)]).unwrap();
    let lhs_parent =
        Tensor::rand_with_seed(&runtime, Dtype::C64, [&space.dual()], [&space], 353_271).unwrap();
    let rhs =
        Tensor::rand_with_seed(&runtime, Dtype::C64, [&space.dual()], [&space], 353_272).unwrap();
    let lhs = lhs_parent.adjoint().unwrap();
    let lhs_shared = lhs.storage_strong_count();

    let lhs_lazy = lhs.compose(&rhs).unwrap();
    let lhs_oracle = old_compose_oracle(&lhs, &rhs);
    assert_c64_close(lhs_lazy.data_c64(), lhs_oracle.data_c64());
    assert_eq!(lhs.storage_strong_count(), lhs_shared);

    let rhs_parent =
        Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space.dual()], 353_273).unwrap();
    let rhs_lazy = rhs_parent.adjoint().unwrap();
    let lhs_shared = lhs.storage_strong_count();
    let rhs_shared = rhs_lazy.storage_strong_count();

    let both_lazy = lhs.compose(&rhs_lazy).unwrap();
    let both_oracle = old_compose_oracle(&lhs, &rhs_lazy);
    assert_c64_close(both_lazy.data_c64(), both_oracle.data_c64());
    assert_eq!(lhs.storage_strong_count(), lhs_shared);
    assert_eq!(rhs_lazy.storage_strong_count(), rhs_shared);
}

#[test]
fn bosonic_u1_and_su2_compose_keep_output_order_and_values() {
    // What: routing all multiplicity-free map composition through the coupled
    // block product preserves asymmetric U1 and non-Abelian SU2 results.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for (fixture, space) in [
        Space::u1([(-2, 1), (0, 2), (1, 3)]),
        Space::su2([(0, 2), (1, 3), (2, 1)]).unwrap(),
    ]
    .into_iter()
    .enumerate()
    {
        let lhs = Tensor::rand_with_seed(
            &runtime,
            Dtype::C64,
            [&space, &space],
            [&space.dual()],
            353_300 + fixture as u64 * 10,
        )
        .unwrap();
        let rhs = Tensor::rand_with_seed(
            &runtime,
            Dtype::C64,
            [&space.dual()],
            [&space, &space],
            353_301 + fixture as u64 * 10,
        )
        .unwrap();

        let composed = lhs.compose(&rhs).unwrap();
        let tensorcontract = lhs.contract(&rhs, &[2], &[0]).unwrap();

        assert_eq!(composed.data_c64(), tensorcontract.data_c64());
        assert_eq!(composed.codomain_rank(), 2);
        assert_eq!(composed.domain_rank(), 2);
        let (_, compose_bytes) = measured_bytes(|| black_box(lhs.compose(&rhs).unwrap()));
        let (_, contract_bytes) = measured_bytes(|| {
            let lhs_axes = (lhs.codomain_rank()..lhs.rank()).collect::<Vec<_>>();
            let rhs_axes = (0..rhs.codomain_rank()).collect::<Vec<_>>();
            black_box(lhs.contract(&rhs, &lhs_axes, &rhs_axes).unwrap())
        });
        assert!(
            compose_bytes <= contract_bytes,
            "fixture {fixture}: compose={compose_bytes} B, contract={contract_bytes} B"
        );
    }
}

#[test]
fn compose_errors_leave_borrowed_operands_unchanged() {
    // What: validation failures occur before composition can mutate or replace
    // either borrowed operand's storage.
    let runtime = Runtime::builder().build().unwrap();
    let foreign_runtime = Runtime::builder().build().unwrap();
    let space = Space::fz2([(0, 2), (1, 2)]).unwrap();
    let lhs =
        Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space.dual()], 353_401).unwrap();
    let rhs =
        Tensor::rand_with_seed(&runtime, Dtype::F64, [&space.dual()], [&space], 353_402).unwrap();
    let foreign = Tensor::rand_with_seed(
        &foreign_runtime,
        Dtype::F64,
        [&space.dual()],
        [&space],
        353_403,
    )
    .unwrap();
    let wrong_shape =
        Tensor::rand_with_seed(&runtime, Dtype::F64, [&space, &space], [&space], 353_404).unwrap();
    let lhs_before = lhs.data().to_vec();
    let rhs_before = rhs.data().to_vec();

    assert!(matches!(lhs.compose(&foreign), Err(Error::RuntimeMismatch)));
    assert!(matches!(
        lhs.compose(&wrong_shape),
        Err(Error::InvalidArgument(_))
    ));
    assert_eq!(lhs.data(), lhs_before);
    assert_eq!(rhs.data(), rhs_before);
}

#[test]
fn direct_compose_does_not_allocate_the_rhs_twist_payload() {
    // What: after both routes are warm, direct map composition saves at least
    // one complete RHS payload versus the former twist-then-contract sequence.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for degeneracy in [12, 24] {
        let space = Space::fz2([(0, degeneracy), (1, degeneracy)]).unwrap();
        let lhs = Tensor::rand_with_seed(
            &runtime,
            Dtype::C64,
            [&space],
            [&space.dual()],
            353_500 + degeneracy as u64,
        )
        .unwrap();
        let rhs = Tensor::rand_with_seed(
            &runtime,
            Dtype::C64,
            [&space.dual()],
            [&space],
            353_600 + degeneracy as u64,
        )
        .unwrap();
        let rhs_payload = rhs.data_c64().len() as u64 * size_of::<Complex64>() as u64;

        black_box(lhs.compose(&rhs).unwrap());
        black_box(old_compose_oracle(&lhs, &rhs));
        let (direct, direct_bytes) = measured_bytes(|| black_box(lhs.compose(&rhs).unwrap()));
        let (oracle, oracle_bytes) = measured_bytes(|| black_box(old_compose_oracle(&lhs, &rhs)));

        assert_c64_close(direct.data_c64(), oracle.data_c64());
        assert!(
            direct_bytes + rhs_payload <= oracle_bytes,
            "degeneracy={degeneracy}: direct={direct_bytes} B, old={oracle_bytes} B, \
             RHS payload={rhs_payload} B"
        );
    }
}
