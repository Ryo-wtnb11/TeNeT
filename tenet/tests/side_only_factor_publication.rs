//! Full QR/LQ/SVD and null spaces of MPS-like sites whose codomain and domain
//! each carry a coupled sector the other side lacks (#1477). Those sectors are
//! identity-completed (MatrixAlgebraKit `one!`), and full-rank sectors drop
//! out of the null spaces, so the factors must stay exactly unitary /
//! complementary on the complete fused spaces.

use std::sync::Arc;

use num_complex::Complex64;
use tenet::core::{
    product_sector, FermionParityFusionRule, PackedProductCodec, ProductFusionRule, SU2FusionRule,
    SU2Irrep, U1FusionRule, U1Irrep, U1SectorLayout, Z2Irrep,
};
use tenet::prelude::{GradedSpace, Runtime, TensorMap};

type Fz2U1Codec = PackedProductCodec<tenet::core::Fz2SectorLayout, U1SectorLayout>;
type Fz2U1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;

macro_rules! assert_close {
    ($actual:expr, $expected:expr) => {{
        let actual = $actual;
        let expected = $expected;
        let error = actual
            .add(expected, 1.0.into(), (-1.0).into())
            .unwrap()
            .norm()
            .unwrap();
        assert!(
            error <= 1e-10 * (1.0 + expected.norm().unwrap()),
            "residual {error}"
        );
    }};
}

/// `values` must not follow a low-order linear recurrence, or the populated
/// sectors lose rank and the null spaces change.
macro_rules! side_only_case {
    ($rule:expr, $left:expr, $phys:expr, $right:expr, $scalar:ty, $value:expr) => {{
        let rt = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new($rule);
        let left = GradedSpace::try_new_with_arc(Arc::clone(&provider), $left).unwrap();
        let phys = GradedSpace::try_new_with_arc(Arc::clone(&provider), $phys).unwrap();
        let right = GradedSpace::try_new_with_arc(Arc::clone(&provider), $right).unwrap();
        let mut counter = 0usize;
        let value = $value;
        let t: TensorMap<_, $scalar> =
            TensorMap::from_block_fn(&rt, [&left, &phys], [&right], |_, _| {
                counter += 1;
                value(counter)
            })
            .unwrap();
        let codomain_id: TensorMap<_, $scalar> = TensorMap::id(&rt, [&left, &phys]).unwrap();
        let domain_id: TensorMap<_, $scalar> = TensorMap::id(&rt, [&right]).unwrap();
        let unitary = |u: &TensorMap<_, $scalar>| {
            let adjoint = u.adjoint().unwrap();
            let inner = TensorMap::id(&rt, u.domain().iter()).unwrap();
            let outer = TensorMap::id(&rt, u.codomain().iter()).unwrap();
            assert_close!(&adjoint.compose(u).unwrap(), &inner);
            assert_close!(&u.compose(&adjoint).unwrap(), &outer);
        };

        let (q, r) = t.qr_full().unwrap();
        assert_close!(&q.compose(&r).unwrap(), &t);
        unitary(&q);
        let (l, q) = t.lq_full().unwrap();
        assert_close!(&l.compose(&q).unwrap(), &t);
        unitary(&q);
        let (u, s, vh) = t.svd_full().unwrap();
        assert_close!(&u.compose(&s).unwrap().compose(&vh).unwrap(), &t);
        unitary(&u);
        unitary(&vh);

        let pseudo = t.pinv(1e-12).unwrap();
        let null = t.left_null().unwrap();
        let null_adjoint = null.adjoint().unwrap();
        assert!(null_adjoint.compose(&t).unwrap().norm().unwrap() <= 1e-10);
        assert_close!(
            &null_adjoint.compose(&null).unwrap(),
            &TensorMap::id(&rt, null.domain().iter()).unwrap()
        );
        assert_close!(
            &null
                .compose(&null_adjoint)
                .unwrap()
                .add(&t.compose(&pseudo).unwrap(), 1.0.into(), 1.0.into())
                .unwrap(),
            &codomain_id
        );
        let null = t.right_null().unwrap();
        let null_adjoint = null.adjoint().unwrap();
        assert!(t.compose(&null_adjoint).unwrap().norm().unwrap() <= 1e-10);
        assert_close!(
            &null.compose(&null_adjoint).unwrap(),
            &TensorMap::id(&rt, null.codomain().iter()).unwrap()
        );
        assert_close!(
            &null_adjoint
                .compose(&null)
                .unwrap()
                .add(&pseudo.compose(&t).unwrap(), 1.0.into(), 1.0.into())
                .unwrap(),
            &domain_id
        );
    }};
}

fn real(k: usize) -> f64 {
    ((k * k) as f64 * 0.37 + k as f64).sin()
}

fn complex(k: usize) -> Complex64 {
    Complex64::new(real(k), ((k * k) as f64 * 0.23 - k as f64).cos())
}

macro_rules! all_symmetries {
    ($scalar:ty, $value:expr) => {{
        // U(1): codomain-only charge 2, domain-only charge 3.
        let q = U1Irrep::new;
        side_only_case!(
            U1FusionRule,
            [(q(0), 2), (q(1), 2)],
            [(q(0), 1), (q(1), 1)],
            [(q(0), 2), (q(1), 3), (q(3), 2)],
            $scalar,
            $value
        );
        // SU(2): codomain-only spin 1, domain-only spin 2.
        let j = SU2Irrep::from_twice_spin;
        side_only_case!(
            SU2FusionRule,
            [(j(1), 2)],
            [(j(1), 1)],
            [(j(0), 1), (j(4), 2)],
            $scalar,
            $value
        );
        // fZ2 x U(1): codomain-only (even, 2), domain-only (odd, 3).
        let f = |odd: bool, charge| {
            product_sector(
                if odd { Z2Irrep::ODD } else { Z2Irrep::EVEN },
                U1Irrep::new(charge),
            )
        };
        side_only_case!(
            Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule),
            [(f(false, 0), 2), (f(true, 1), 2)],
            [(f(false, 0), 1), (f(true, 1), 1)],
            [(f(false, 0), 2), (f(true, 1), 3), (f(true, 3), 2)],
            $scalar,
            $value
        );
    }};
}

#[test]
fn side_only_full_and_null_factors_are_complete_real() {
    all_symmetries!(f64, real);
}

#[test]
fn side_only_full_and_null_factors_are_complete_complex() {
    all_symmetries!(Complex64, complex);
}
