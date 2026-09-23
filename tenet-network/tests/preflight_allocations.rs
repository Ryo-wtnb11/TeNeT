//! Allocation contracts of the network metadata preflight (#1371).
//!
//! The counter is per thread, so the tests do not disturb each other; each
//! call under test runs on the test's own thread.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use tenet::core::{U1FusionRule, U1Irrep};
use tenet::prelude::Runtime;
use tenet::typed::{GradedSpace, TensorMap};
use tenet_network::{
    slice_plan_for, static_network_operand_preflight, tensor, DenseCostModel, DenseTensorInfo,
    GreedyDenseOptimizer, Network, NetworkIR, SlicedPlan, StaticTopologySpec, TemporaryLabel,
};

#[path = "../../tenet/tests/braiding_probe/mod.rs"]
mod braiding_probe;
use braiding_probe::{ProbeSector, RealBraidingProbe};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

fn record() {
    let _ = COUNTING.try_with(|counting| {
        if counting.get() {
            let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        }
    });
}

// SAFETY: every call forwards to `System` with the caller's arguments; the
// counter touches only const-initialised thread-locals, which never allocate.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record();
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Allocation calls (alloc and realloc) made by `run` on this thread.
fn allocations<T>(run: impl FnOnce() -> T) -> (u64, T) {
    ALLOCATIONS.with(|count| count.set(0));
    COUNTING.with(|counting| counting.set(true));
    let value = run();
    COUNTING.with(|counting| counting.set(false));
    (ALLOCATIONS.with(Cell::get), value)
}

/// `a: [v, v; w]`, `b: [w; v, v]` and `c: [w*; v, v]`, 8 U(1) sectors of
/// degeneracy 2 per leg.
fn u1_operands(runtime: &Runtime) -> [TensorMap<U1FusionRule, f64>; 3] {
    let provider = Arc::new(U1FusionRule);
    let space = |shift: i32| {
        GradedSpace::try_new_with_arc(
            Arc::clone(&provider),
            (0..8).map(|charge| (U1Irrep::new(charge - 4 + shift), 2)),
        )
        .unwrap()
    };
    let (v, w) = (space(0), space(1));
    [
        TensorMap::rand_with_seed(runtime, [&v, &v], [&w], 1_371_100).unwrap(),
        TensorMap::rand_with_seed(runtime, [&w], [&v, &v], 1_371_101).unwrap(),
        TensorMap::rand_with_seed(runtime, [&w.try_dual().unwrap()], [&v, &v], 1_371_102).unwrap(),
    ]
}

static PAIR: StaticTopologySpec = StaticTopologySpec {
    inputs: &[&["i", "j", "m"], &["m", "k", "l"]],
    conj: &[false, false],
    codomain_splits: &[Some(2), Some(1)],
    output: &["i", "j", "k", "l"],
    output_codomain_rank: Some(2),
    contracted: &[&[None, None, None], &[Some((0, 2)), None, None]],
};

/// `conj(b)` is `[v, v; w]`, so it contracts with `b` over all three legs.
static CONJ_PAIR: StaticTopologySpec = StaticTopologySpec {
    inputs: &[&["m", "i", "j"], &["m", "i", "j"]],
    conj: &[true, false],
    codomain_splits: &[Some(1), Some(1)],
    output: &[],
    output_codomain_rank: Some(0),
    contracted: &[
        &[None, None, None],
        &[Some((0, 0)), Some((0, 1)), Some((0, 2))],
    ],
};

/// `PAIR` with no precomputed pairing, which a runtime `Network` and a traced
/// lowering also have: the preflight must fall back to the label scan rather
/// than skip the space check.
static UNPAIRED: StaticTopologySpec = StaticTopologySpec {
    contracted: &[],
    ..PAIR
};

/// A pairing that puts both ends of a pair inside one `conj` operand. That
/// is a trace, which `tensor!` lowers through `StaticTrace`s instead, and the
/// written order the pairing is in runs opposite to the lowered order the
/// preflight walks a `conj` operand in, so its endpoints are not the scan's.
static INTRA_OPERAND: StaticTopologySpec = StaticTopologySpec {
    inputs: &[&["i", "j", "m"], &["t", "t", "m"]],
    conj: &[false, true],
    codomain_splits: &[Some(2), Some(1)],
    output: &["i", "j"],
    output_codomain_rank: Some(2),
    contracted: &[&[None, None, None], &[None, Some((1, 0)), Some((0, 2))]],
};

/// `PAIR` with an endpoint outside its own pairing, which would panic if it
/// were indexed.
static OUT_OF_RANGE: StaticTopologySpec = StaticTopologySpec {
    contracted: &[&[None, None, None], &[Some((0, 7)), None, None]],
    ..PAIR
};

/// #1394: a spec whose pairing does not describe its operands is not trusted.
/// Skipping a space check is a silent wrong answer, so the preflight falls
/// back to the scan and still rejects `c`'s `m` leg.
#[test]
fn a_spec_without_a_pairing_still_checks_every_contracted_leg() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let [a, b, c] = u1_operands(&runtime);
    static_network_operand_preflight(&[&a, &b], &UNPAIRED).unwrap();
    let error = static_network_operand_preflight(&[&a, &c], &UNPAIRED).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("space mismatch for contracted label `m`"),
        "{error}"
    );

    // An endpoint outside the pairing is dropped the same way, instead of
    // panicking where it would be indexed.
    static_network_operand_preflight(&[&a, &b], &OUT_OF_RANGE).unwrap();
    let error = static_network_operand_preflight(&[&a, &c], &OUT_OF_RANGE).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("space mismatch for contracted label `m`"),
        "{error}"
    );

    // A pair inside one `conj` operand: the preflight walks that operand's
    // axes in the opposite order, so trusting the pairing would pair the
    // mirror axes — the `debug_assert!` fires if this one is trusted. It
    // scans instead, and still rejects `conj(b)`'s `m` leg, whose duality is
    // the one `b` contracts with, not `conj(b)`.
    let error = static_network_operand_preflight(&[&a, &b], &INTRA_OPERAND).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("space mismatch for contracted label `m`"),
        "{error}"
    );
}

/// #1371: the preflight a warm plan-cache hit runs on every call allocates
/// nothing, with and without a `conj` operand, and still rejects a leg of the
/// wrong duality. It would allocate if it built dualised spaces, label lists
/// or a label map.
#[test]
fn the_warm_preflight_allocates_nothing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let [a, b, c] = u1_operands(&runtime);
    // Any lazily created provider or runtime state is built here, not below.
    static_network_operand_preflight(&[&a, &b], &PAIR).unwrap();
    static_network_operand_preflight(&[&b, &b], &CONJ_PAIR).unwrap();

    let (count, result) = allocations(|| static_network_operand_preflight(&[&a, &b], &PAIR));
    assert!(result.is_ok());
    assert_eq!(count, 0, "untraced preflight");
    let (count, result) = allocations(|| static_network_operand_preflight(&[&b, &b], &CONJ_PAIR));
    assert!(result.is_ok());
    assert_eq!(count, 0, "conj preflight");

    // Non-vacuity: the same check rejects `c`, whose `m` leg has the wrong
    // duality, and allocates only for that error.
    let error = static_network_operand_preflight(&[&a, &c], &PAIR).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("space mismatch for contracted label `m`"),
        "{error}"
    );

    // Whole warm call, reported for the review ledger (not asserted: it is
    // dominated by the contraction itself).
    for _ in 0..3 {
        drop(tensor!([i, j; k, l] = a[i, j; m] * b[m; k, l]).unwrap());
    }
    let (count, output) = allocations(|| tensor!([i, j; k, l] = a[i, j; m] * b[m; k, l]));
    drop(output.unwrap());
    eprintln!("warm tensor! call: {count} allocation calls");
}

/// #1371 / #1372 review P2-1: `PlannedNetwork::execute` and the sliced path
/// reject a non-symmetric braiding before their first step or accumulator:
/// with the typed `contract`'s error, allocating no more than the typed
/// `contract` does to reject the same operands (the error itself).
#[test]
fn planned_and_sliced_executions_reject_non_symmetric_braiding_first() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(RealBraidingProbe::<true>, [(ProbeSector, 2)]).unwrap();
    let lhs = TensorMap::<_, f64>::rand_with_seed(&runtime, [&leg], [&leg], 1_371_200).unwrap();
    let rhs = TensorMap::<_, f64>::rand_with_seed(&runtime, [&leg], [&leg], 1_371_201).unwrap();
    let tensors = [&lhs, &rhs];
    let labels = |names: &[&str]| {
        names
            .iter()
            .copied()
            .map(TemporaryLabel::from)
            .collect::<Vec<_>>()
    };
    let inputs = vec![labels(&["a", "x"]), labels(&["x", "b"])];
    let output = labels(&["a", "b"]);
    let network = Network::new(
        inputs.clone(),
        vec![false; 2],
        vec![Some(1), Some(1)],
        output.clone(),
        Some(1),
    )
    .unwrap();
    let planned = network.plan(&tensors, &GreedyDenseOptimizer).unwrap();
    let ir = NetworkIR::from_labels(inputs, output).unwrap();
    let cost = DenseCostModel::from_network(
        &ir,
        &[
            DenseTensorInfo::new(vec![2, 2]),
            DenseTensorInfo::new(vec![2, 2]),
        ],
    )
    .unwrap();
    let dense = SlicedPlan::new(
        planned.plan().clone(),
        slice_plan_for(&ir, planned.plan(), &cost, &labels(&["a"])),
    );
    let sliced = network
        .lower_symmetric_sliced_plan(&tensors, dense)
        .unwrap();

    let (contract_count, contract) =
        allocations(|| lhs.contract(&rhs, &[1], &[0], &[0, 1]).map(drop));
    let contract = contract.unwrap_err().to_string();

    let (count, error) = allocations(|| planned.execute(&tensors).map(drop));
    assert_eq!(error.unwrap_err().to_string(), contract, "execute");
    assert!(
        count <= contract_count,
        "execute: {count} > {contract_count}"
    );

    let (count, error) = allocations(|| {
        network
            .execute_symmetric_sliced(&tensors, sliced, usize::MAX)
            .map(drop)
    });
    let error = error.unwrap_err();
    assert!(
        matches!(
            &error,
            tenet_network::SymmetricSliceExecutionError::Tensor(inner)
                if inner.to_string() == contract
        ),
        "{error:?}"
    );
    assert!(
        count <= contract_count,
        "sliced: {count} > {contract_count}"
    );
}
