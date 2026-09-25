//! Warm eager `contract` with a non-identity `output_axes` costs no more than
//! `contract` in the default order followed by `permute` (#1461), and returns
//! the same tensor. TensorKit `blas_contract!` takes the same shape of work
//! when the output order is not the GEMM's: a temporary plus a permuting
//! `tensoradd!` (its `copyC` path).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::Mutex;

use num_complex::{Complex32, Complex64};
use tenet::prelude::*;

#[path = "../../tests/support/numerics.rs"]
mod numerics;

struct CountingAllocator;

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

fn record(size: usize) {
    if ENABLED.get() {
        CALLS.set(CALLS.get() + 1);
        BYTES.set(BYTES.get() + size);
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(new_size);
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;
// Why serialize: process-global layout caches are shared across test threads.
static MEASUREMENT_LOCK: Mutex<()> = Mutex::new(());

/// (calls, bytes) allocated while `f` runs; the result is dropped afterwards.
fn allocations<T>(f: impl FnOnce() -> T) -> ((usize, usize), T) {
    CALLS.set(0);
    BYTES.set(0);
    ENABLED.set(true);
    let value = f();
    ENABLED.set(false);
    ((CALLS.get(), BYTES.get()), value)
}

/// Warm cost and value of `contract(lhs_axes, rhs_axes, output_axes)` against
/// the default order followed by `permute(codomain, domain)`.
macro_rules! assert_no_costlier_than_permute {
    ($runtime:expr, $a:expr, $b:expr, $lhs_axes:expr, $rhs_axes:expr, $output_axes:expr,
     $codomain:expr, $domain:expr $(,)?) => {{
        let (lhs_axes, rhs_axes, output_axes): (&[usize], &[usize], &[usize]) =
            ($lhs_axes, $rhs_axes, $output_axes);
        let (codomain, domain): (&[usize], &[usize]) = ($codomain, $domain);
        let identity: Vec<usize> = (0..output_axes.len()).collect();
        let fused = || {
            black_box($a)
                .contract($b, lhs_axes, rhs_axes, output_axes)
                .unwrap()
        };
        let separate = || {
            black_box($a)
                .contract($b, lhs_axes, rhs_axes, &identity)
                .unwrap()
                .permute(codomain, domain)
                .unwrap()
        };
        fused();
        separate();
        let misses = $runtime.tree_transform_cache_info().misses();
        let (fused_cost, fused_value) = allocations(fused);
        let (separate_cost, separate_value) = allocations(separate);

        // What: the warm call compiles no new transform plan.
        assert_eq!($runtime.tree_transform_cache_info().misses(), misses);
        // What: the same tensor, bit for bit.
        assert_eq!(fused_value.data(), separate_value.data());
        assert_eq!(fused_value.codomain_rank(), codomain.len());
        // What: no more allocation calls or bytes than the two-step route.
        assert!(
            fused_cost.0 <= separate_cost.0 && fused_cost.1 <= separate_cost.1,
            "contract({lhs_axes:?}, {rhs_axes:?}, {output_axes:?}) {fused_cost:?} vs \
             contract + permute {separate_cost:?}"
        );
    }};
}

/// `a: V⊗V ← W⊗W` with `W = V` or `W = V*`, `b: W⊗W ← V`, `m: V ← V`.
/// Contracting `a`'s whole domain with `b`'s whole codomain is the core-GEMM
/// form (dual `W` adds the fermionic supertrace twist); `a`'s last leg with
/// `m` is not. Both output orders braid an open leg of the right operand into
/// the codomain.
macro_rules! assert_output_axes_cost {
    ($provider:expr, $sectors:expr $(,)?) => {{
        let _guard = MEASUREMENT_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let v = GradedSpace::try_new($provider, $sectors.into_iter().map(|s| (s, 3))).unwrap();
        for w in [v.clone(), v.try_dual().unwrap()] {
            let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&w, &w], 1).unwrap();
            let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&w, &w], [&v], 2).unwrap();
            let m = TensorMap::<_, f64>::rand_with_seed(&runtime, [&w], [&w], 3).unwrap();
            let x = TensorMap::<_, f64>::rand_with_seed(&runtime, [&w, &w], [&v, &v], 4).unwrap();
            // What: the value matches an independent route. Pre-permuting the
            // left operand to `[0, 2] <- [1, 3]` puts its contracted legs at
            // `[1, 3]`, off the core form, so the one-call source-transform
            // route computes it (TensorKit's copyA). Covered for an owned and
            // a lazy-adjoint left operand. Bound: an entry is bilinear in the
            // operands, so `len(lhs) * len(rhs)` terms.
            for lhs in [a.clone(), x.adjoint().unwrap()] {
                let got = lhs.contract(&b, &[2, 3], &[0, 1], &[0, 2, 1]).unwrap();
                let want = lhs
                    .permute(&[0, 2], &[1, 3])
                    .unwrap()
                    .contract(&b, &[1, 3], &[0, 1], &[0, 2, 1])
                    .unwrap();
                assert_eq!(got.codomain_rank(), want.codomain_rank());
                assert_eq!(got.leg_dims().unwrap(), want.leg_dims().unwrap());
                numerics::assert_slices_close(
                    "contract(output_axes) vs pre-permuted operand",
                    got.data(),
                    want.data(),
                    a.data().len() * b.data().len(),
                );
            }
            assert_no_costlier_than_permute!(
                runtime,
                &a,
                &b,
                &[2, 3],
                &[0, 1],
                &[0, 2, 1],
                &[0, 2],
                &[1],
            );
            assert_no_costlier_than_permute!(
                runtime,
                &a,
                &m,
                &[3],
                &[0],
                &[0, 3, 1, 2],
                &[0, 3, 1],
                &[2],
            );
        }
    }};
}

fn centered(count: i32) -> impl Iterator<Item = i32> {
    (0..count).map(move |i| i - (count - 1) / 2)
}

#[test]
fn warm_output_axes_u1_costs_no_more_than_contract_then_permute() {
    assert_output_axes_cost!(U1FusionRule, centered(3).map(U1Irrep::new));
}

#[test]
fn warm_output_axes_su2_costs_no_more_than_contract_then_permute() {
    assert_output_axes_cost!(SU2FusionRule, (0..3).map(SU2Irrep::from_twice_spin));
}

#[test]
fn warm_output_axes_fz2_u1_costs_no_more_than_contract_then_permute() {
    let parity = |q: i32| {
        if q.rem_euclid(2) == 0 {
            Z2Irrep::EVEN
        } else {
            Z2Irrep::ODD
        }
    };
    assert_output_axes_cost!(
        ProductFusionRule::<FermionParityFusionRule, U1FusionRule>::new(
            FermionParityFusionRule,
            U1FusionRule
        ),
        centered(3).map(|q| ProductSector::new(parity(q), U1Irrep::new(q))),
    );
}
