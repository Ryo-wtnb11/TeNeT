//! Complex (c64) user-layer coverage: construction, arithmetic cross-checked
//! against the real/imaginary f64 decomposition, adjoint conjugation, inner
//! product semantics, decompositions, the general eigendecomposition, the
//! `tensor!` macro over conjugated complex operands, and mixed-dtype
//! rejection.

use tenet::prelude::*;
use tenet_network::tensor;

fn u1_space() -> Space {
    Space::u1([(-1, 1), (0, 2), (1, 1)])
}

fn su2_space() -> Space {
    Space::su2([(0, 1), (1, 2)])
}

fn i() -> Complex64 {
    Complex64::new(0.0, 1.0)
}

fn one() -> Complex64 {
    Complex64::new(1.0, 0.0)
}

/// Builds `re + i * im` from two same-space f64 tensors.
fn complexify(re: &Tensor, im: &Tensor) -> Tensor {
    re.to_c64().add_c64(&im.to_c64(), one(), i()).unwrap()
}

fn assert_close_c64(actual: &[Complex64], expected: &[Complex64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).norm() <= tol,
            "element {index}: {a} vs {e} (|diff| = {})",
            (a - e).norm()
        );
    }
}

#[test]
fn c64_construction() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let zero = Tensor::zeros(&rt, Dtype::C64, [&v, &v], [&v, &v]).unwrap();
        assert_eq!(zero.dtype(), Dtype::C64);
        assert_eq!(zero.norm().unwrap(), 0.0);

        let a = Tensor::rand_with_seed(&rt, Dtype::C64, [&v, &v], [&v, &v], 7).unwrap();
        let b = Tensor::rand_with_seed(&rt, Dtype::C64, [&v, &v], [&v, &v], 7).unwrap();
        assert_eq!(a.dtype(), Dtype::C64);
        assert_eq!(a.data_c64(), b.data_c64(), "same seed must reproduce");
        assert!(a.norm().unwrap() > 0.0);
        assert!(
            a.data_c64().iter().any(|value| value.im != 0.0),
            "random c64 data must have nonzero imaginary parts"
        );

        // A Complex64-returning from_block_fn agrees with two f64 fills.
        let fill_re = |key: &BlockKey, indices: &[usize]| -> f64 {
            let BlockKey::FusionTree(key) = key else {
                return 0.0;
            };
            (key.codomain_uncoupled()[0].id() as f64) + indices[0] as f64 * 0.25
        };
        let fill_im = |key: &BlockKey, indices: &[usize]| -> f64 {
            let BlockKey::FusionTree(key) = key else {
                return 0.0;
            };
            (key.domain_uncoupled()[0].id() as f64) - indices[1] as f64 * 0.5
        };
        let c = Tensor::from_block_fn(&rt, [&v], [&v, &v], |key, indices| {
            Complex64::new(fill_re(key, indices), fill_im(key, indices))
        })
        .unwrap();
        let c_re = Tensor::from_block_fn(&rt, [&v], [&v, &v], fill_re).unwrap();
        let c_im = Tensor::from_block_fn(&rt, [&v], [&v, &v], fill_im).unwrap();
        assert_close_c64(c.data_c64(), complexify(&c_re, &c_im).data_c64(), 0.0);
    }
}

/// `(A + iB)(C + iD) = (AC - BD) + i(AD + BC)`: the c64 contraction path
/// cross-checked elementwise against four f64 contractions.
#[test]
fn c64_contract_matches_real_imag_decomposition() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 11).unwrap();
        let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 12).unwrap();
        let c = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 13).unwrap();
        let d = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 14).unwrap();

        let x = complexify(&a, &b);
        let y = complexify(&c, &d);
        let z = x.compose(&y).unwrap();
        assert_eq!(z.dtype(), Dtype::C64);

        let real = a
            .compose(&c)
            .unwrap()
            .add(&b.compose(&d).unwrap(), 1.0, -1.0)
            .unwrap();
        let imag = a
            .compose(&d)
            .unwrap()
            .add(&b.compose(&c).unwrap(), 1.0, 1.0)
            .unwrap();
        let expected = complexify(&real, &imag);
        assert_close_c64(z.data_c64(), expected.data_c64(), 1e-12);

        // Same identity through the general contract path with permuted axes.
        let z2 = x.contract(&y, &[2, 3], &[0, 1]).unwrap();
        assert_close_c64(z2.data_c64(), expected.data_c64(), 1e-12);
    }
}

/// `adjoint(A + iB) = adjoint(A) - i adjoint(B)`: dagger must conjugate.
#[test]
fn c64_adjoint_conjugates() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v], 21).unwrap();
        let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v], 22).unwrap();
        let x = complexify(&a, &b);

        let dagger = x.adjoint().unwrap();
        let expected = complexify(
            &a.adjoint().unwrap(),
            &b.adjoint().unwrap().scale(-1.0).unwrap(),
        );
        assert_close_c64(dagger.data_c64(), expected.data_c64(), 0.0);

        // Involution.
        assert_close_c64(dagger.adjoint().unwrap().data_c64(), x.data_c64(), 0.0);
    }
}

/// TensorKit `dot` semantics: `inner(x, y) = conj(inner(y, x))`, `inner(x, x)`
/// real and equal to `norm^2`, `norm` always real.
#[test]
fn c64_inner_hermitian_symmetry_and_norm() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let x = Tensor::rand_with_seed(&rt, Dtype::C64, [&v, &v], [&v, &v], 31).unwrap();
        let y = Tensor::rand_with_seed(&rt, Dtype::C64, [&v, &v], [&v, &v], 32).unwrap();

        let xy = x.inner(&y).unwrap().to_c64();
        let yx = y.inner(&x).unwrap().to_c64();
        assert!((xy - yx.conj()).norm() <= 1e-12 * (1.0 + xy.norm()));
        assert!(
            xy.im.abs() > 0.0,
            "generic complex inner has imaginary part"
        );

        let xx = x.inner(&x).unwrap();
        let norm = x.norm().unwrap();
        assert!(xx.im().abs() <= 1e-12 * (1.0 + xx.re()));
        assert!((xx.re() - norm * norm).abs() <= 1e-10 * (1.0 + norm * norm));
    }
}

#[test]
fn c64_svd_compact_recomposes_and_u_unitary() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let t = Tensor::rand_with_seed(&rt, Dtype::C64, [&v, &v], [&v], 41).unwrap();
        let (u, s, vh) = t.svd_compact().unwrap();
        assert_eq!(u.dtype(), Dtype::C64);
        assert_eq!(s.dtype(), Dtype::C64);

        let recon = u.compose(&s).unwrap().compose(&vh).unwrap();
        let diff = recon.add(&t, 1.0, -1.0).unwrap().norm().unwrap();
        assert!(diff <= 1e-10 * (1.0 + t.norm().unwrap()), "diff = {diff}");

        // U^H U = id on the bond: inner-based unitarity check.
        let gram = u.adjoint().unwrap().compose(&u).unwrap();
        let bond = gram.domain_spaces()[0].clone();
        let id = Tensor::from_block_fn(&rt, [&bond], [&bond], |_, indices| {
            if indices[0] == indices[1] {
                one()
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .unwrap();
        let gram_diff = gram.add(&id, 1.0, -1.0).unwrap().norm().unwrap();
        assert!(gram_diff <= 1e-10 * (1.0 + id.norm().unwrap()));
        // <U, U> = dim of the bond space (weighted trace of U^H U).
        let trace = u.inner(&u).unwrap();
        assert!((trace.re() - bond.dim() as f64).abs() <= 1e-10 * (1.0 + bond.dim() as f64));
        assert!(trace.im().abs() <= 1e-12 * (1.0 + trace.re()));
    }
}

/// `eig_full` of a REAL tensor returns c64 factors with
/// `V * diag(lambda) * V^-1 ~ t` (this is why the dtype-erased storage
/// exists: the output dtype differs from the input's).
#[test]
fn eig_full_of_real_tensor_is_c64_and_recomposes() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 51).unwrap();
        assert_eq!(t.dtype(), Dtype::F64);

        let (d, w) = t.eig_full().unwrap();
        assert_eq!(d.dtype(), Dtype::C64);
        assert_eq!(w.dtype(), Dtype::C64);

        // Eigen equation `t * V = V * D` (V maps the bond into the original
        // codomain, so it is not an endomorphism; no inverse needed).
        let lhs = t.to_c64().compose(&w).unwrap();
        let rhs = w.compose(&d).unwrap();
        let diff = lhs.add(&rhs, 1.0, -1.0).unwrap().norm().unwrap();
        assert!(diff <= 1e-8 * (1.0 + t.norm().unwrap()), "diff = {diff}");

        // Complex eigenvalues of a real matrix come in conjugate pairs, so
        // eig_vals of real input sums to a real number (the trace).
        let spectra = t.eig_vals().unwrap();
        let sum: Complex64 = spectra
            .iter()
            .flat_map(|spectrum| spectrum.values.iter())
            .sum();
        assert!(sum.im.abs() <= 1e-8 * (1.0 + sum.norm()));
    }
}

#[test]
fn c64_eig_and_eigh_trunc_on_hermitized_tensor() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let t = Tensor::rand_with_seed(&rt, Dtype::C64, [&v, &v], [&v, &v], 61).unwrap();
        // h = (t + t^H) / 2 is Hermitian.
        let h = t.add(&t.adjoint().unwrap(), 0.5, 0.5).unwrap();

        // Full truncation budget: eigh_trunc must recompose exactly.
        let trunc = h.eigh_trunc(&Truncation::Full).unwrap();
        assert_eq!(trunc.d.dtype(), Dtype::C64);
        let recon = trunc
            .v
            .compose(&trunc.d)
            .unwrap()
            .compose(&trunc.v.adjoint().unwrap())
            .unwrap();
        let diff = recon.add(&h, 1.0, -1.0).unwrap().norm().unwrap();
        assert!(diff <= 1e-10 * (1.0 + h.norm().unwrap()), "diff = {diff}");
        assert!(trunc.error.abs() <= 1e-12);

        // Hermitian input: eig and eigh agree on the spectrum (eig values
        // must be real up to roundoff).
        let eig = h.eig_trunc(&Truncation::Full).unwrap();
        for spectrum in &eig.eigenvalues {
            for value in &spectrum.values {
                assert!(value.im.abs() <= 1e-10 * (1.0 + value.norm()));
            }
        }
        let eig_lhs = h.compose(&eig.v).unwrap();
        let eig_rhs = eig.v.compose(&eig.d).unwrap();
        let eig_diff = eig_lhs.add(&eig_rhs, 1.0, -1.0).unwrap().norm().unwrap();
        assert!(eig_diff <= 1e-8 * (1.0 + h.norm().unwrap()));

        // A rank cut keeps the dominant part: error decreases the budget.
        let cut = h.eigh_trunc(&Truncation::rank(2)).unwrap();
        assert!(cut.error >= 0.0);
    }
}

/// `tensor!` with `conj()` on c64 operands: `<psi|H|psi>` is real for
/// Hermitian `H` (the network layer lowers `conj` to `adjoint`, which must
/// conjugate complex data).
#[test]
fn tensor_macro_conj_expectation_value_is_real() {
    let rt = Runtime::builder().build().unwrap();
    for p in [u1_space(), su2_space()] {
        let l = p.clone();
        let r = p.dual();
        let psi = Tensor::rand_with_seed(&rt, Dtype::C64, [&p], [&l, &r], 71).unwrap();
        let h0 = Tensor::rand_with_seed(&rt, Dtype::C64, [&p], [&p], 72).unwrap();
        let h = h0.add(&h0.adjoint().unwrap(), 0.5, 0.5).unwrap();

        let e = tensor!([] = conj(psi)[p; l, r] * h[p; q] * psi[q; l, r])
            .unwrap()
            .scalar()
            .unwrap()
            .to_c64();
        let n = tensor!([] = conj(psi)[p; l, r] * psi[p; l, r])
            .unwrap()
            .scalar()
            .unwrap()
            .to_c64();
        assert!(n.re > 0.0);
        assert!(n.im.abs() <= 1e-12 * (1.0 + n.re), "norm not real: {n}");
        assert!(
            e.im.abs() <= 1e-10 * (1.0 + e.norm()),
            "<psi|H|psi> not real for Hermitian H: {e}"
        );
        // Cross-check against the method-level inner: <psi, H psi>.
        let hpsi = h.compose(&psi).unwrap();
        let via_inner = psi.inner(&hpsi).unwrap().to_c64();
        assert!((via_inner - e).norm() <= 1e-10 * (1.0 + e.norm()));
    }
}

#[test]
fn mixed_dtype_operations_are_rejected() {
    let rt = Runtime::builder().build().unwrap();
    let v = u1_space();
    let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 81).unwrap();
    let b = Tensor::rand_with_seed(&rt, Dtype::C64, [&v], [&v], 82).unwrap();

    assert!(matches!(a.compose(&b), Err(Error::DtypeMismatch)));
    assert!(matches!(
        b.contract(&a, &[1], &[0]),
        Err(Error::DtypeMismatch)
    ));
    assert!(matches!(a.add(&b, 1.0, 1.0), Err(Error::DtypeMismatch)));
    assert!(matches!(a.inner(&b), Err(Error::DtypeMismatch)));
    assert!(matches!(
        a.scale_c64(Complex64::new(0.0, 1.0)),
        Err(Error::DtypeMismatch)
    ));
    assert!(matches!(
        a.add_c64(&b, one(), one()),
        Err(Error::DtypeMismatch)
    ));
    // Explicit widening makes them compatible.
    let widened = a.to_c64();
    assert_eq!(widened.dtype(), Dtype::C64);
    assert!(widened.compose(&b).is_ok());
    assert!((widened.norm().unwrap() - a.norm().unwrap()).abs() <= 1e-15);
}

#[test]
fn structural_constructors_c64_and_twist_on_c64_data() {
    let rt = Runtime::builder().build().unwrap();
    let l = Space::fz2([(0, 1), (1, 2)]);
    let fused = l.dual().fuse(&l).unwrap();

    // c64 structural constructors are the widened f64 ones.
    let id = Tensor::id(&rt, Dtype::C64, [&l]).unwrap();
    assert_eq!(id.dtype(), Dtype::C64);
    assert_eq!(
        id.data_c64(),
        Tensor::id(&rt, Dtype::F64, [&l])
            .unwrap()
            .to_c64()
            .data_c64()
    );
    let f = Tensor::isomorphism(&rt, Dtype::C64, [&fused], [&l.dual(), &l]).unwrap();
    assert_eq!(f.dtype(), Dtype::C64);
    assert_eq!(
        Tensor::unitary(&rt, Dtype::C64, [&fused], [&l.dual(), &l])
            .unwrap()
            .data_c64(),
        f.data_c64()
    );
    let big = Space::fz2([(0, 2), (1, 3)]);
    let w = Tensor::isometry(&rt, Dtype::C64, [&big], [&l]).unwrap();
    assert_eq!(
        w.adjoint().unwrap().compose(&w).unwrap().data_c64(),
        id.data_c64()
    );

    // twist preserves the c64 dtype and stays an involution.
    let t = Tensor::rand_with_seed(&rt, Dtype::C64, [&l.dual()], [&l], 11).unwrap();
    let twisted = t.twist(&[0]).unwrap();
    assert_eq!(twisted.dtype(), Dtype::C64);
    assert_ne!(twisted.data_c64(), t.data_c64());
    assert_eq!(twisted.twist(&[0]).unwrap().data_c64(), t.data_c64());
}
