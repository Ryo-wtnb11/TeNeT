//! Complex user-level contracts not already owned by the typed operation and
//! decomposition suites.

use std::sync::Arc;

use tenet::core::{
    CheckedFusionAlgebra, FusionAlgebraError, MultiplicityFreeAdmissionMode,
    MultiplicityFreeRigidSymbols, SU2FusionRule, SU2Irrep, SectorCodec, TypedSectorAdmission,
    U1FusionRule, U1Irrep,
};
use tenet::prelude::{Complex64, Runtime};
use tenet::typed::{GradedSpace, TensorMap};
use tenet_network::tensor;

fn i() -> Complex64 {
    Complex64::new(0.0, 1.0)
}

fn one() -> Complex64 {
    Complex64::new(1.0, 0.0)
}

fn complexify<R>(re: &TensorMap<R, f64>, im: &TensorMap<R, f64>) -> TensorMap<R, Complex64>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    re.to_c64().add(&im.to_c64(), one(), i()).unwrap()
}

fn assert_close(actual: &[Complex64], expected: &[Complex64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).norm() <= tolerance,
            "element {index}: {actual} vs {expected}"
        );
    }
}

fn assert_complex_contract_identity<R>(runtime: &Runtime, space: &GradedSpace<R>)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    let a =
        TensorMap::<R, f64>::rand_with_seed(runtime, [space, space], [space, space], 11).unwrap();
    let b =
        TensorMap::<R, f64>::rand_with_seed(runtime, [space, space], [space, space], 12).unwrap();
    let c =
        TensorMap::<R, f64>::rand_with_seed(runtime, [space, space], [space, space], 13).unwrap();
    let d =
        TensorMap::<R, f64>::rand_with_seed(runtime, [space, space], [space, space], 14).unwrap();

    let x = complexify(&a, &b);
    let y = complexify(&c, &d);
    let real = a
        .compose(&c)
        .unwrap()
        .add(&b.compose(&d).unwrap(), 1.0, -1.0)
        .unwrap();
    let imaginary = a
        .compose(&d)
        .unwrap()
        .add(&b.compose(&c).unwrap(), 1.0, 1.0)
        .unwrap();
    let expected = complexify(&real, &imaginary);

    assert_close(x.compose(&y).unwrap().data(), expected.data(), 1.0e-12);
    assert_close(
        x.contract(&y, &[2, 3], &[0, 1], &[0, 1, 2, 3])
            .unwrap()
            .data(),
        expected.data(),
        1.0e-12,
    );
}

/// `(A + iB)(C + iD) = (AC - BD) + i(AD + BC)` through both typed
/// composition and arbitrary-axis contraction.
#[test]
fn c64_contract_matches_real_imag_decomposition() {
    let runtime = Runtime::builder().build().unwrap();
    let u1 = GradedSpace::try_new_shared(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 1),
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    assert_complex_contract_identity(&runtime, &u1);

    let su2 = GradedSpace::try_new_shared(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    assert_complex_contract_identity(&runtime, &su2);
}

fn assert_typed_macro_conj<R>(runtime: &Runtime, p: &GradedSpace<R>)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
{
    let l = p.clone();
    let r = p.try_dual().unwrap();
    let psi = TensorMap::<R, Complex64>::rand_with_seed(runtime, [p], [&l, &r], 71).unwrap();
    let h0 = TensorMap::<R, Complex64>::rand_with_seed(runtime, [p], [p], 72).unwrap();
    let h = h0
        .add(&h0.adjoint().unwrap(), one() * 0.5, one() * 0.5)
        .unwrap();

    let expectation = tensor!([] = conj(psi)[p; l, r] * h[p; q] * psi[q; l, r])
        .unwrap()
        .scalar()
        .unwrap();
    let norm = tensor!([] = conj(psi)[p; l, r] * psi[p; l, r])
        .unwrap()
        .scalar()
        .unwrap();
    assert!(norm.re > 0.0);
    assert!(norm.im.abs() <= 1.0e-12 * (1.0 + norm.re));
    assert!(expectation.im.abs() <= 1.0e-10 * (1.0 + expectation.norm()));

    let via_inner = psi.inner(&h.compose(&psi).unwrap()).unwrap();
    assert!((via_inner - expectation).norm() <= 1.0e-10 * (1.0 + expectation.norm()));
}

/// The network layer lowers `conj` to typed adjoint, including complex
/// conjugation: a Hermitian expectation value is real.
#[test]
fn tensor_macro_conj_expectation_value_is_real() {
    let runtime = Runtime::builder().build().unwrap();
    let u1 = GradedSpace::try_new_shared(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 1),
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    assert_typed_macro_conj(&runtime, &u1);

    let su2 = GradedSpace::try_new_shared(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    assert_typed_macro_conj(&runtime, &su2);
}
