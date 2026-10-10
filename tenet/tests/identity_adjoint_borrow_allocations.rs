//! #2147 performance contract: on the DynamicTree route, a lazy adjoint whose
//! own tree transform is the identity reaches the core GEMM as its parent's
//! borrowed blocks with an adjoint op, not as an operand-sized copy
//! (TensorKit `cfaa073e` `blas_contract!`, `tensoroperations.jl:383-450`:
//! `mul!` over `block(A, c)'` when `A'` needs no `copyA`; QSpace `dd2cc7e1`
//! `wbarray.cc:3472-3482`: `conj` folded into the GEMM `'C'` op).
//!
//! Fixture: `a = p'` with `p: [V, V] ← [V]` contracted over its whole domain
//! against legs 0 and 2 of `u: [V, X, V] ← []`, so `u` is permuted and the
//! route is DynamicTree while `a` is already in core form. `|p| ≠ |u|`, so an
//! allocation of exactly `|p|` elements is the copy of `a` and one of `|u|`
//! elements is the (still required) copy of `u`, the positive control.

use std::hint::black_box;
use std::sync::Arc;

use tenet::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use tenet::typed::{ContractSpec, GradedSpace, Runtime, TensorMap};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

const SPEC: ContractSpec<'static> = ContractSpec {
    lhs: &[1, 2],
    rhs: &[0, 2],
    codomain: &[0],
    domain: &[1],
};

/// Asserts, at two degeneracy scales, that one call on a trimmed runtime
/// allocates nothing of exactly `|p|` elements and at least one buffer of
/// `|u|` elements; `$spaces(scale)` gives `(V, X)`.
macro_rules! assert_borrowed {
    ($name:expr, $spaces:expr) => {{
        let _serial = counting_alloc::serial();
        for scale in [1, 4] {
            let (v, x) = $spaces(scale);
            let runtime = Runtime::builder().dense_threads(1).build().unwrap();
            let value = |_: &_, index: &[usize]| 0.25 + index.iter().sum::<usize>() as f64 * 0.125;
            let p: TensorMap<_, f64> =
                TensorMap::from_subblock_fn(&runtime, [&v, &v], [&v], value).unwrap();
            let u: TensorMap<_, f64> =
                TensorMap::from_subblock_fn(&runtime, [&v, &x, &v], [], value).unwrap();
            let bytes = |t: &TensorMap<_, f64>| {
                t.subblocks()
                    .unwrap()
                    .map(|(_, view)| view.shape().iter().product::<usize>())
                    .sum::<usize>()
                    * std::mem::size_of::<f64>()
            };
            let (p_bytes, u_bytes) = (bytes(&p), bytes(&u));
            assert_ne!(p_bytes, u_bytes);
            let a = p.adjoint().unwrap();
            let call = || {
                black_box(a.contract(&u, &SPEC).unwrap());
            };
            call();
            call();
            runtime.trim_host_contract_scratch();
            let ((), p_allocs) = counting_alloc::measure_matching(p_bytes..=p_bytes, call);
            let retained = runtime.host_contract_scratch_bytes();
            runtime.trim_host_contract_scratch();
            let ((), u_allocs) = counting_alloc::measure_matching(u_bytes..=u_bytes, call);
            let (p_copies, u_copies) = (p_allocs.matched_calls, u_allocs.matched_calls);
            eprintln!(
                "{} x{scale}: |p| {p_bytes} B copied {p_copies}x, |u| {u_bytes} B copied \
                 {u_copies}x; call {} allocs {} B peak {} B; retained scratch {retained} B",
                $name, p_allocs.calls, p_allocs.bytes, p_allocs.peak_live_bytes,
            );
            // What: no allocation the size of `a`'s payload, while the
            // permuted `u` is still copied into its own scratch (the probe
            // sees operand copies).
            assert_eq!(p_copies, 0, "{} x{scale}", $name);
            assert!(u_copies >= 1, "{} x{scale}", $name);
        }
    }};
}

#[test]
fn su2_identity_adjoint_is_borrowed_on_the_dynamic_tree_route() {
    assert_borrowed!("SU(2)", |scale: usize| {
        let space = |degs: [usize; 3]| {
            GradedSpace::try_new(
                Arc::new(SU2FusionRule),
                [0, 1, 2]
                    .into_iter()
                    .zip(degs)
                    .map(|(twice, deg)| (SU2Irrep::from_twice_spin(twice), deg * scale)),
            )
            .unwrap()
        };
        (space([2, 3, 2]), space([1, 1, 3]))
    });
}

#[test]
fn u1_identity_adjoint_is_borrowed_on_the_dynamic_tree_route() {
    assert_borrowed!("U(1)", |scale: usize| {
        let space = |degs: [usize; 3]| {
            GradedSpace::try_new(
                Arc::new(U1FusionRule),
                [-1, 0, 1]
                    .into_iter()
                    .zip(degs)
                    .map(|(charge, deg)| (U1Irrep::new(charge), deg * scale)),
            )
            .unwrap()
        };
        (space([2, 3, 2]), space([1, 1, 3]))
    });
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_su2_identity_adjoint_is_borrowed_on_the_dynamic_tree_route() {
    assert_borrowed!("checked SU(2)", |scale: usize| {
        let space = |degs: [usize; 3]| {
            GradedSpace::try_new(
                Arc::new(tenet::sector::SUNFusionRule::new(2).unwrap()),
                [0i64, 1, 2]
                    .into_iter()
                    .zip(degs)
                    .map(|(twice, deg)| (vec![twice], deg * scale)),
            )
            .unwrap()
        };
        (space([2, 3, 2]), space([1, 1, 3]))
    });
}
