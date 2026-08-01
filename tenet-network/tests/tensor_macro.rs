//! Integration tests for the `tensor!` macro and the `Network` planner +
//! executor over the user-layer `Tensor`.

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
    Space::su2([(0, 2), (1, 2), (2, 1)]).unwrap()
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

/// SU(3) remains on the private erased macro executor until macro cutover;
/// keep its multi-step crossed-output orientation pinned meanwhile.
#[test]
fn su3_multistep_orientation_matches_manual_contraction() {
    let runtime = Runtime::builder().build().unwrap();
    let space = Space::su3([((1, 0), 2), ((0, 1), 1)]).unwrap();
    let a = Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space], 224_811).unwrap();
    let b = Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space], 224_812).unwrap();
    let c = Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space], 224_813).unwrap();

    let actual = tensor!([l; i] = a[i; j] * b[j; k] * c[k; l]).unwrap();
    let expected = a
        .contract(&b, &[1], &[0])
        .unwrap()
        .contract(&c, &[1], &[0])
        .unwrap()
        .permute(&[1], &[0])
        .unwrap();
    assert!(actual
        .data_c64()
        .iter()
        .zip(expected.data_c64())
        .all(|(lhs, rhs)| (*lhs - *rhs).norm() < 1e-12));
}

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
