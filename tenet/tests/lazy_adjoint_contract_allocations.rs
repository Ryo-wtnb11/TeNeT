//! Warm eager `contract`/`compose` with a lazy-adjoint receiver (#1414).
//!
//! The adjoint-oriented tree-pair plan lives in the Runtime-owned transform
//! store, and the operand projection reuses the parent's memoized adjoint, so
//! a warm lazy-adjoint call allocates no more than the owned `contract` (#1419).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::Mutex;

use tenet::prelude::*;

struct CountingAllocator;

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && ENABLED.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() && ENABLED.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;
// Why serialize: process-global layout caches are shared across test threads,
// so a concurrent test's evictions would move this thread's warm counts.
static MEASUREMENT_LOCK: Mutex<()> = Mutex::new(());

fn allocations<T>(f: impl FnOnce() -> T) -> (u64, T) {
    ALLOCATIONS.set(0);
    ENABLED.set(true);
    let value = f();
    ENABLED.set(false);
    (ALLOCATIONS.get(), value)
}

/// Warm `contract_conj` may exceed the owned `contract` by at most this many
/// allocation calls. It was 5 to 8 before #1419 (the per-call adjoint HomSpace,
/// storage map, axis-check Vecs and, for SU2, the contracted HomSpace compare);
/// it is now 0 on these rows. Why not exact pins: allocator and std
/// differences move the absolute counts between platforms by the same offset.
const LAZY_ADJOINT_SLACK: u64 = 0;
/// Upper bound on warm `contract_conj` allocation calls. Recompiling the
/// oriented plan per call cost 474 or more on every E1 row.
const LAZY_ADJOINT_CEILING: u64 = 64;

/// `A: V⊗V ← V⊗V` with three sectors of degeneracy four (the E1 `r4_s3_d4`
/// row). The Runtime transform store must see no miss on a warm call.
macro_rules! assert_warm_lazy_adjoint {
    ($provider:expr, $sectors:expr $(,)?) => {{
        let _guard = MEASUREMENT_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let leg = GradedSpace::try_new($provider, $sectors.into_iter().map(|s| (s, 4))).unwrap();
        let a =
            TensorMap::<_, f64>::rand_with_seed(&runtime, [&leg, &leg], [&leg, &leg], 1).unwrap();
        let a2 =
            TensorMap::<_, f64>::rand_with_seed(&runtime, [&leg, &leg], [&leg, &leg], 2).unwrap();
        let matrix = TensorMap::<_, f64>::rand_with_seed(&runtime, [&leg], [&leg], 4).unwrap();
        let lazy = a.adjoint().unwrap();
        let open = [0, 1, 2, 3];
        let contract_conj = || {
            black_box(&lazy)
                .contract(&matrix, &[0], &[1], &open)
                .unwrap()
        };
        let compose_conj = || black_box(&lazy).compose(&a2).unwrap();
        let contract = || black_box(&a).contract(&matrix, &[0], &[1], &open).unwrap();

        let cold_contract = contract_conj();
        let cold_compose = compose_conj();
        contract();
        let misses = runtime.tree_transform_cache_info().misses();
        let (contract_conj_calls, warm_contract) = allocations(contract_conj);
        let (compose_conj_calls, warm_compose) = allocations(compose_conj);
        let (contract_calls, _) = allocations(contract);
        // What: the first warm call is already the steady state.
        assert_eq!(allocations(contract_conj).0, contract_conj_calls);
        assert_eq!(allocations(compose_conj).0, compose_conj_calls);

        // What: warm calls reuse every Runtime-owned transform plan.
        assert_eq!(runtime.tree_transform_cache_info().misses(), misses);
        // What: warm replay is deterministic.
        assert_eq!(warm_contract.data(), cold_contract.data());
        assert_eq!(warm_compose.data(), cold_compose.data());
        // What: a warm lazy-adjoint contract allocates like the owned one.
        assert!(
            contract_conj_calls <= contract_calls + LAZY_ADJOINT_SLACK,
            "contract_conj {contract_conj_calls} vs owned contract {contract_calls}"
        );
        assert!(
            contract_conj_calls <= LAZY_ADJOINT_CEILING,
            "contract_conj {contract_conj_calls}"
        );
    }};
}

fn centered(count: i32) -> impl Iterator<Item = i32> {
    (0..count).map(move |i| i - (count - 1) / 2)
}

#[test]
fn warm_lazy_adjoint_u1_allocates_like_owned_contract() {
    assert_warm_lazy_adjoint!(U1FusionRule, centered(3).map(U1Irrep::new));
}

#[test]
fn warm_lazy_adjoint_su2_allocates_like_owned_contract() {
    assert_warm_lazy_adjoint!(SU2FusionRule, (0..3).map(SU2Irrep::from_twice_spin));
}

#[test]
fn warm_lazy_adjoint_fz2_u1_allocates_like_owned_contract() {
    let parity = |q: i32| {
        if q.rem_euclid(2) == 0 {
            Z2Irrep::EVEN
        } else {
            Z2Irrep::ODD
        }
    };
    assert_warm_lazy_adjoint!(
        ProductFusionRule::<FermionParityFusionRule, U1FusionRule>::new(
            FermionParityFusionRule,
            U1FusionRule
        ),
        centered(3).map(|q| ProductSector::new(parity(q), U1Irrep::new(q))),
    );
}
