//! Integration tests for the `tensor!` macro and the `Network` planner +
//! executor over the user-layer `Tensor`.

use tenet::prelude::*;
use tenet_network::{
    tensor, GreedyDenseOptimizer, LabelOrderDenseOptimizer, Network, NetworkExecutionWorkspace,
    TemporaryLabel, TensorId,
};

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

#[test]
fn pairwise_macro_matches_direct_contract() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 101).unwrap();
        let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 102).unwrap();

        let c = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
        let expected = a.contract(&b, &[2, 3], &[0, 1]).unwrap();
        assert_close(c.data(), expected.data(), 1e-12);
        assert_eq!(c.codomain_rank(), 2);
        assert_eq!(c.domain_rank(), 2);
    }
}

#[test]
fn macro_accepts_parenthesized_operand_expressions() {
    let rt = Runtime::builder().build().unwrap();
    let v = u1_space();
    let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 103).unwrap();
    let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 104).unwrap();
    let pair = (a, b);

    let c = tensor!([i, j; m, n] = (pair.0)[i, j; k, l] * (pair.1)[k, l; m, n]).unwrap();
    let expected = pair.0.contract(&pair.1, &[2, 3], &[0, 1]).unwrap();
    assert_close(c.data(), expected.data(), 1e-12);
}

#[test]
fn permuted_output_labels_match_contract_ordered() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 111).unwrap();
        let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 112).unwrap();

        let c = tensor!([j, i; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
        let expected = a
            .contract_ordered(&b, &[2, 3], &[0, 1], &[1, 0, 2, 3])
            .unwrap();
        assert_close(c.data(), expected.data(), 1e-12);
    }
}

#[test]
fn single_tensor_macro_is_a_permute() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 121).unwrap();
        let p = tensor!([j; i] = t[i; j]).unwrap();
        let expected = t.permute(&[1], &[0]).unwrap();
        assert_close(p.data(), expected.data(), 1e-12);
    }
}

#[test]
fn scalar_output_with_conj_matches_norm_squared() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 131).unwrap();
        let n2 = tensor!([] = conj(a)[i, j; k, l] * a[i, j; k, l])
            .unwrap()
            .scalar()
            .unwrap()
            .try_f64()
            .unwrap();
        let norm = a.norm().unwrap();
        assert!(
            (n2 - norm * norm).abs() <= 1e-10 * (1.0 + norm * norm),
            "<a|a> = {n2} but norm^2 = {}",
            norm * norm
        );
    }
}

/// The psi-H-psi energy shape: `<psi| H |psi>` as a 3-tensor network with a
/// conjugated bra, cross-checked against a manual two-step contraction.
#[test]
fn three_tensor_chain_with_conj_matches_manual_contraction() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let psi = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v, &v], 141).unwrap();
        let h = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 142).unwrap();

        let e = tensor!([] = conj(psi)[p; l, r] * h[p; q] * psi[q; l, r])
            .unwrap()
            .scalar()
            .unwrap()
            .try_f64()
            .unwrap();

        // Manual: m1 = H |psi> with legs (p; l, r), then close with the bra.
        // adjoint(psi) has flat legs (l*, r*; p*): domain legs lead.
        let m1 = h.contract(&psi, &[1], &[0]).unwrap();
        let bra = psi.adjoint().unwrap();
        let manual = bra
            .contract(&m1, &[2, 0, 1], &[0, 1, 2])
            .unwrap()
            .scalar()
            .unwrap()
            .try_f64()
            .unwrap();

        assert!(
            (e - manual).abs() <= 1e-10 * (1.0 + manual.abs()),
            "macro energy {e} vs manual {manual}"
        );
    }
}

/// 4-tensor chain where greedy planning differs from naive left-to-right:
/// results identical, greedy's estimated cost strictly lower, and the first
/// greedy step is the cheap tail pair rather than the head pair.
#[test]
fn greedy_order_beats_naive_left_to_right_on_a_chain() {
    let rt = Runtime::builder().build().unwrap();
    let dim = |d: usize| Space::u1([(0, d)]);
    let (va, vb, vc, vd, ve) = (dim(4), dim(8), dim(4), dim(2), dim(2));
    let t1 = Tensor::rand_with_seed(&rt, Dtype::F64, [&va], [&vb], 151).unwrap();
    let t2 = Tensor::rand_with_seed(&rt, Dtype::F64, [&vb], [&vc], 152).unwrap();
    let t3 = Tensor::rand_with_seed(&rt, Dtype::F64, [&vc], [&vd], 153).unwrap();
    let t4 = Tensor::rand_with_seed(&rt, Dtype::F64, [&vd], [&ve], 154).unwrap();
    let tensors = [&t1, &t2, &t3, &t4];

    let label = |s: &str| TemporaryLabel::from(s);
    let network = Network::new(
        vec![
            vec![label("a"), label("b")],
            vec![label("b"), label("c")],
            vec![label("c"), label("d")],
            vec![label("d"), label("e")],
        ],
        vec![false; 4],
        vec![None; 4],
        vec![label("a"), label("e")],
        None,
    )
    .unwrap();

    let greedy = network.plan(&tensors, &GreedyDenseOptimizer).unwrap();
    // Naive left-to-right = contract labels in written order b, c, d.
    let naive = network
        .plan(
            &tensors,
            &LabelOrderDenseOptimizer::new(vec![label("b"), label("c"), label("d")]),
        )
        .unwrap();

    // Greedy starts with the cheap tail pair (t3, t4), not (t1, t2).
    let first = &greedy.plan().steps()[0];
    assert_eq!(
        (first.lhs(), first.rhs()),
        (TensorId::new(2), TensorId::new(3)),
        "greedy should contract the cheap (c,d)x(d,e) pair first"
    );
    let naive_first = &naive.plan().steps()[0];
    assert_eq!(
        (naive_first.lhs(), naive_first.rhs()),
        (TensorId::new(0), TensorId::new(1))
    );

    assert!(
        greedy.plan().total_cost() < naive.plan().total_cost(),
        "greedy cost {} should beat naive cost {}",
        greedy.plan().total_cost(),
        naive.plan().total_cost()
    );

    let from_greedy = greedy.execute(&tensors).unwrap();
    let from_naive = naive.execute(&tensors).unwrap();
    assert_close(from_greedy.data(), from_naive.data(), 1e-12);

    // The macro (greedy default) agrees too.
    let from_macro = tensor!([a; e] = t1[a; b] * t2[b; c] * t3[c; d] * t4[d; e]).unwrap();
    assert_close(from_macro.data(), from_greedy.data(), 1e-12);
}

/// Reusing one workspace across warm executions preserves the compiled
/// schedule's numerical result, including intermediate orientation.
#[test]
fn planned_network_reuses_execution_workspace() {
    let rt = Runtime::builder().build().unwrap();
    let v = u1_space();
    let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 157).unwrap();
    let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 158).unwrap();
    let c = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 159).unwrap();
    let labels = |names: &[&str]| {
        names
            .iter()
            .map(|name| TemporaryLabel::from(*name))
            .collect()
    };
    let network = Network::new(
        vec![
            labels(&["i", "j"]),
            labels(&["j", "k"]),
            labels(&["k", "l"]),
        ],
        vec![false; 3],
        vec![None; 3],
        labels(&["l", "i"]),
        Some(1),
    )
    .unwrap();
    let tensors = [&a, &b, &c];
    let planned = network.plan(&tensors, &GreedyDenseOptimizer).unwrap();
    let expected = planned.execute(&tensors).unwrap();
    let mut workspace = NetworkExecutionWorkspace::default();

    let first = planned
        .execute_with_workspace(&tensors, &mut workspace)
        .unwrap();
    let second = planned
        .execute_with_workspace(&tensors, &mut workspace)
        .unwrap();
    assert_close(first.data(), expected.data(), 1e-12);
    assert_close(second.data(), expected.data(), 1e-12);
}

/// A written `;` split that contradicts the tensor's codomain rank is a
/// runtime error (labels are checked at compile time, spaces at run time).
#[test]
fn wrong_input_codomain_split_is_rejected() {
    let rt = Runtime::builder().build().unwrap();
    let v = u1_space();
    let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 161).unwrap();
    let u = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 162).unwrap();
    // t is [v] <- [v] (codomain rank 1) but written as [i, j; ].
    let result = tensor!([i; k] = t[i, j;] * u[j; k]);
    assert!(matches!(result, Err(Error::InvalidArgument(_))));
}

/// A compiled schedule rejects a tensor whose rank still matches but whose
/// codomain/domain orientation no longer matches the topology it lowered.
#[test]
fn planned_network_rejects_codomain_orientation_drift() {
    let rt = Runtime::builder().build().unwrap();
    let v = u1_space();
    let planned_input = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 165).unwrap();
    let network = Network::new(
        vec![vec![TemporaryLabel::from("i"), TemporaryLabel::from("j")]],
        vec![false],
        vec![None],
        vec![TemporaryLabel::from("i"), TemporaryLabel::from("j")],
        Some(1),
    )
    .unwrap();
    let planned = network
        .plan(&[&planned_input], &GreedyDenseOptimizer)
        .unwrap();

    let drifted = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [], 166).unwrap();
    let error = planned.execute(&[&drifted]).unwrap_err();
    assert!(
        matches!(&error, Error::InvalidArgument(message) if message.contains("topology drifted")),
        "unexpected error: {error}"
    );
}

/// Contracted legs are validated structurally at plan time: same sectors AND
/// same per-sector degeneracies (mutually dual spaces). A degeneracy mismatch
/// is rejected with a message naming the label and both legs' content.
#[test]
fn contracted_leg_degeneracy_mismatch_is_rejected_with_both_legs_spelled_out() {
    let rt = Runtime::builder().build().unwrap();
    let v = u1_space();
    let w = Space::u1([(-1, 2), (0, 4), (1, 2)]); // charge 0 degeneracy differs
    let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 163).unwrap();
    let u = Tensor::rand_with_seed(&rt, Dtype::F64, [&w], [&w], 164).unwrap();
    let err = tensor!([i; k] = t[i; j] * u[j; k]).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("space mismatch for contracted label `j`"),
        "unexpected message: {message}"
    );
    assert!(
        message.contains("mutually dual"),
        "unexpected message: {message}"
    );
}

/// Field-access operands parse without parentheses: `svd.u[...]`,
/// `pair.0[...]`, and `conj(svd.u)[...]` all work and agree with the
/// parenthesized spelling.
#[test]
fn field_access_operands_parse_and_contract() {
    let rt = Runtime::builder().build().unwrap();
    let v = u1_space();
    let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v], 401).unwrap();
    let svd = t.svd_trunc(&Truncation::Full).unwrap();

    // svd.u : [v, v] <- [bond], svd.vh : [bond] <- [v].
    let bare = tensor!([i, j; m] = svd.u[i, j; k] * svd.s[k; l] * svd.vh[l; m]).unwrap();
    let parens = tensor!([i, j; m] = (svd.u)[i, j; k] * (svd.s)[k; l] * (svd.vh)[l; m]).unwrap();
    assert_close(bare.data(), parens.data(), 1e-15);
    assert_close(bare.data(), t.data(), 1e-10);

    // conj() around a field-access chain, reducing to the norm.
    let n2 = tensor!([] = conj(svd.u)[i, j; k] * svd.u[i, j; k])
        .unwrap()
        .scalar()
        .unwrap()
        .try_f64()
        .unwrap();
    let norm = svd.u.norm().unwrap();
    assert!((n2 - norm * norm).abs() <= 1e-10 * (1.0 + norm * norm));

    // Tuple-index fields.
    let qr = t.qr_compact().unwrap();
    let recomposed = tensor!([i, j; m] = qr.0[i, j; k] * qr.1[k; m]).unwrap();
    assert_close(recomposed.data(), t.data(), 1e-10);
}
