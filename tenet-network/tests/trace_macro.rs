//! Partial traces in the `tensor!` macro: intra-operand repeated labels
//! lower to the user-layer categorical trace (`Tensor::trace_pairs`), which
//! runs the expert `tensortrace_fusion` path. The expert path itself is
//! oracle-tested against TensorKit in
//! `tenet-tensors/src/tests/tensortrace.rs`
//! (`tensortrace_fusion_fermion_parity_matches_tensorkit_supertrace`,
//! `tensortrace_fusion_su2_includes_quantum_dimension_factor`); here we
//! verify the macro/network lowering agrees with it and, independently, with
//! contraction against an identity map (twist-free rules) and analytic
//! block sums.

use tenet::prelude::*;
use tenet_network::tensor;

fn assert_close(lhs: &[f64], rhs: &[f64], tol: f64) {
    assert_eq!(lhs.len(), rhs.len(), "data lengths differ");
    for (index, (a, b)) in lhs.iter().zip(rhs).enumerate() {
        assert!(
            (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs())),
            "element {index} differs: {a} vs {b}"
        );
    }
}

fn u1_space() -> Space {
    Space::u1([(-1, 2), (0, 3), (1, 2)])
}

fn su2_space() -> Space {
    Space::su2([(0, 2), (1, 2), (2, 1)])
}

fn fz2_space() -> Space {
    Space::fz2([(0, 2), (1, 3)])
}

/// Identity endomorphism on `v`.
fn eye(rt: &Runtime, v: &Space) -> Tensor {
    Tensor::from_block_fn(
        rt,
        [v],
        [v],
        |_, idx| {
            if idx[0] == idx[1] {
                1.0
            } else {
                0.0
            }
        },
    )
    .unwrap()
}

/// `tensor!` partial trace over one pair equals both the user-layer trace
/// primitive (expert tensortrace lowering) elementwise, for U1, SU2 and
/// fermionic fZ2 (supertrace semantics), f64 and c64.
#[test]
fn partial_trace_matches_trace_pairs_elementwise() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space(), fz2_space()] {
        let w = v.clone();
        let vd = v.dual();
        let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &vd], [&w], 201).unwrap();

        let traced = tensor!([; j] = a[i, i; j]).unwrap();
        let expected = a.trace_pairs(&[(0, 1)]).unwrap();
        assert_close(traced.data(), expected.data(), 1e-12);
        assert_eq!(traced.codomain_rank(), 0);
        assert_eq!(traced.domain_rank(), 1);

        let a_c = Tensor::rand_with_seed(&rt, Dtype::C64, [&v, &vd], [&w], 202).unwrap();
        let traced_c = tensor!([; j] = a_c[i, i; j]).unwrap();
        let expected_c = a_c.trace_pairs(&[(0, 1)]).unwrap();
        let diff: f64 = traced_c
            .data_c64()
            .iter()
            .zip(expected_c.data_c64())
            .map(|(x, y)| (x - y).norm())
            .sum();
        assert!(diff <= 1e-12, "c64 partial trace differs by {diff}");
    }
}

/// Independent oracle for twist-free rules (U1, SU2): the partial trace
/// equals pairwise contraction with an identity endomorphism.
#[test]
fn partial_trace_matches_identity_contraction_for_twist_free_rules() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let w = v.clone();
        let vd = v.dual();
        let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &vd], [&w], 211).unwrap();
        let id = eye(&rt, &v);

        let traced = tensor!([; j] = a[i, i; j]).unwrap();
        let via_identity = tensor!([; j] = a[i, k; j] * id[k; i]).unwrap();
        assert_close(traced.data(), via_identity.data(), 1e-12);
    }
}

/// Full trace of an identity endomorphism = the quantum dimension of the
/// space (TensorKit `tr(id(V)) == dim(V)`), checking the SU2
/// quantum-dimension factors through the macro path.
#[test]
fn full_trace_of_identity_is_quantum_dimension() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let id = eye(&rt, &v);
        let trace = tensor!([] = id[i; i])
            .unwrap()
            .scalar()
            .unwrap()
            .try_f64()
            .unwrap();
        assert!(
            (trace - v.dim() as f64).abs() <= 1e-12,
            "tr(id) = {trace}, dim = {}",
            v.dim()
        );
        // TensorKit-named tr() agrees.
        let tr = id.tr().unwrap().to_c64();
        assert!((tr - Complex64::new(trace, 0.0)).norm() <= 1e-12);
    }
}

/// What: an fZ2 full tensor-macro trace is the SUPERTRACE (even block trace
/// minus odd block trace), matching the expert-path TensorKit oracle test
/// `tensortrace_fusion_fermion_parity_matches_tensorkit_supertrace`.
/// Diagonal entries here: even block diag(2, 3), odd block diag(5, 6, 7),
/// so the macro gives -13 while ordinary `Tensor::tr` gives 23.
#[test]
fn fz2_macro_trace_is_supertrace_while_tensor_tr_is_ordinary() {
    let rt = Runtime::builder().build().unwrap();
    let v = fz2_space();
    let t = Tensor::from_block_fn(&rt, [&v], [&v], |key, idx| {
        if idx[0] != idx[1] {
            return 9.0; // off-diagonal noise must not contribute
        }
        match key {
            BlockKey::FusionTree(key) if key.codomain_uncoupled()[0].id() == 0 => {
                2.0 + idx[0] as f64
            }
            _ => 5.0 + idx[0] as f64,
        }
    })
    .unwrap();

    let trace = tensor!([] = t[i; i])
        .unwrap()
        .scalar()
        .unwrap()
        .try_f64()
        .unwrap();
    assert!((trace - (-13.0)).abs() <= 1e-12, "supertrace = {trace}");
    let tr = t.tr().unwrap().to_c64();
    assert!((tr - Complex64::new(23.0, 0.0)).norm() <= 1e-12);
}

/// Trace and pairwise contraction combined in one expression, against the
/// manual two-step (trace_pairs, then contract).
#[test]
fn trace_and_contract_combined_matches_manual_two_step() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space(), fz2_space()] {
        let w = v.clone();
        let vd = v.dual();
        let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &vd], [&w], 221).unwrap();
        let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&w], [&w], 222).unwrap();

        let combined = tensor!([; m] = a[i, i; j] * b[j; m]).unwrap();
        let manual = a
            .trace_pairs(&[(0, 1)])
            .unwrap()
            .contract(&b, &[0], &[0])
            .unwrap();
        assert_close(combined.data(), manual.data(), 1e-12);
    }
}

/// conj() on a traced operand: @tensor conj(a)[i, i] is the trace of a's
/// adjoint (conjugated trace for c64 data).
#[test]
fn conj_operand_partial_trace_matches_adjoint_trace() {
    let rt = Runtime::builder().build().unwrap();
    let v = u1_space();
    let a = Tensor::rand_with_seed(&rt, Dtype::C64, [&v], [&v], 231).unwrap();

    let traced = tensor!([] = conj(a)[i; i])
        .unwrap()
        .scalar()
        .unwrap()
        .to_c64();
    let expected = a.adjoint().unwrap().tr().unwrap().to_c64();
    assert!(
        (traced - expected).norm() <= 1e-12,
        "conj trace {traced} vs adjoint trace {expected}"
    );
    // And it is the conjugate of the plain trace.
    let plain = a.tr().unwrap().to_c64();
    assert!((traced - plain.conj()).norm() <= 1e-12);
}

/// Multiple trace pairs on one operand reduce to rank 0 in one step.
#[test]
fn two_trace_pairs_reduce_to_scalar() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), fz2_space()] {
        let vd = v.dual();
        let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 241).unwrap();
        // Legs: (v, v; v, v): flat outward spaces (v, v, v*, v*);
        // pairs (0, 2) and (1, 3) are mutually dual.
        let via_macro = tensor!([] = a[i, j; i, j])
            .unwrap()
            .scalar()
            .unwrap()
            .try_f64()
            .unwrap();
        let via_pairs = a
            .trace_pairs(&[(0, 2), (1, 3)])
            .unwrap()
            .scalar()
            .unwrap()
            .try_f64()
            .unwrap();
        assert!(
            (via_macro - via_pairs).abs() <= 1e-12 * (1.0 + via_pairs.abs()),
            "{via_macro} vs {via_pairs}"
        );
        let _ = vd;
    }
}

/// tr() rejects non-endomorphisms; trace_pairs rejects bad axis lists.
#[test]
fn trace_error_paths() {
    let rt = Runtime::builder().build().unwrap();
    let v = u1_space();
    let w = Space::u1([(0, 4)]);
    let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&w], 251).unwrap();
    assert!(matches!(t.tr(), Err(Error::InvalidArgument(_))));

    let e = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 252).unwrap();
    assert!(matches!(
        e.trace_pairs(&[(0, 0)]),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        e.trace_pairs(&[(0, 5)]),
        Err(Error::InvalidArgument(_))
    ));
}
