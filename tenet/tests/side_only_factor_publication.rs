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

/// TensorKit `initialize_output(qr_full!/lq_full!)`: Q of full QR maps
/// `fuse(codomain)` into the codomain, Q of full LQ maps the domain into
/// `fuse(domain)`, so every side-only sector is identity-completed (#1524).
/// `$qr_bond`/`$lq_bond` are those fused spaces built by `GradedSpace::fuse`,
/// independently of the factorization.
#[cfg(feature = "racah-generated")]
macro_rules! full_qr_lq_bond_case {
    ($rt:expr, $scalar:ty, $codomain:expr, $domain:expr, $qr_bond:expr, $lq_bond:expr, $seed:expr) => {{
        let rt = $rt;
        let owned_adjoint = |u: &TensorMap<_, $scalar>| {
            let adjoint = u.adjoint().unwrap();
            adjoint.add(&adjoint, 1.0.into(), 0.0.into()).unwrap()
        };
        // Checked Generic has no `id`: a Hermitian map is the identity iff
        // every eigenvalue is one, and a count equal to the expected reduced
        // dimension rules out an unstored (zero) side-only block.
        let identity = |gram: TensorMap<_, $scalar>, reduced: usize| {
            let values = gram
                .eigh_vals()
                .unwrap()
                .into_iter()
                .flat_map(|spectrum| spectrum.values)
                .collect::<Vec<_>>();
            assert_eq!(values.len(), reduced);
            assert!(
                values.iter().all(|value| (value - 1.0).abs() <= 1e-10),
                "{values:?}"
            );
        };
        let reduced = |space: &GradedSpace<_>| space.degeneracies().iter().sum::<usize>();
        let unitary = |u: &TensorMap<_, $scalar>, bond: &GradedSpace<_>| {
            let adjoint = owned_adjoint(u);
            identity(adjoint.compose(u).unwrap(), reduced(bond));
            identity(u.compose(&adjoint).unwrap(), reduced(bond));
        };
        let t: TensorMap<_, $scalar> =
            TensorMap::rand_with_seed(rt, $codomain, $domain, $seed).unwrap();
        let (q, r) = t.qr_full().unwrap();
        assert_eq!(q.domain(), std::slice::from_ref($qr_bond));
        assert_eq!(r.codomain(), q.domain());
        assert_close!(&q.compose(&r).unwrap(), &t);
        unitary(&q, $qr_bond);
        let (l, q) = t.lq_full().unwrap();
        assert_eq!(q.codomain(), std::slice::from_ref($lq_bond));
        assert_eq!(l.domain(), q.codomain());
        assert_close!(&l.compose(&q).unwrap(), &t);
        unitary(&q, $lq_bond);
        // Null spaces already use the side-aware builder: a side-only sector
        // is wholly null, so rank + null dimension is the fused dimension.
        let rank = t
            .svd_vals()
            .unwrap()
            .into_iter()
            .flat_map(|spectrum| spectrum.values)
            .filter(|&value| value > 1e-10)
            .count();
        let n = t.left_null().unwrap();
        let n_adjoint = owned_adjoint(&n);
        assert!(n_adjoint.compose(&t).unwrap().norm().unwrap() <= 1e-10);
        identity(n_adjoint.compose(&n).unwrap(), reduced($qr_bond) - rank);
        let n = t.right_null().unwrap();
        let n_adjoint = owned_adjoint(&n);
        assert!(t.compose(&n_adjoint).unwrap().norm().unwrap() <= 1e-10);
        identity(n.compose(&n_adjoint).unwrap(), reduced($lq_bond) - rank);
    }};
}

/// The multiplicity-free comparison fixture: U(1) with codomain-only charge
/// 2 and domain-only charge 3 obeys the same output-space rule.
#[cfg(feature = "racah-generated")]
macro_rules! multiplicity_free_full_qr_lq_bonds {
    ($d:ty) => {{
        let rt = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let q = U1Irrep::new;
        let space = |irreps: Vec<(U1Irrep, usize)>| {
            GradedSpace::try_new_with_arc(Arc::clone(&provider), irreps).unwrap()
        };
        let left = space(vec![(q(0), 2), (q(1), 2)]);
        let phys = space(vec![(q(0), 1), (q(1), 1)]);
        let right = space(vec![(q(0), 2), (q(1), 3), (q(3), 2)]);
        let fused = left.fuse(&phys).unwrap();
        full_qr_lq_bond_case!(&rt, $d, [&left, &phys], [&right], &fused, &right, 1524);
        full_qr_lq_bond_case!(&rt, $d, [&right], [&left, &phys], &right, &fused, 1525);
    }};
}

#[cfg(feature = "racah-generated")]
mod checked_generic {
    use super::*;
    use tenet::typed::SUNFusionRule;

    /// SU(3), `a = 8^2 + 1`, `b = 8^2 + 1 + 6^2`: `a ⊗ a` holds 8 with
    /// multiplicity two and the codomain-only 10, 10̄, 27; `b` holds the
    /// domain-only 6. Both orientations put side-only sectors on each side.
    macro_rules! checked_full_qr_lq_bonds {
        ($d:ty) => {{
            let rt = Runtime::builder().dense_threads(1).build().unwrap();
            let provider = Arc::new(SUNFusionRule::new(3).unwrap());
            let space = |irreps: Vec<(Vec<i64>, usize)>| {
                GradedSpace::try_new_with_arc(Arc::clone(&provider), irreps).unwrap()
            };
            let a = space(vec![(vec![1, 1], 2), (vec![0, 0], 1)]);
            let b = space(vec![(vec![1, 1], 2), (vec![0, 0], 1), (vec![2, 0], 2)]);
            let fused = a.fuse(&a).unwrap();
            full_qr_lq_bond_case!(&rt, $d, [&a, &a], [&b], &fused, &b, 1526);
            full_qr_lq_bond_case!(&rt, $d, [&b], [&a, &a], &b, &fused, 1527);
        }};
    }

    #[test]
    fn checked_generic_full_qr_lq_bonds_are_fused_sides() {
        multiplicity_free_full_qr_lq_bonds!(f64);
        checked_full_qr_lq_bonds!(f64);
        multiplicity_free_full_qr_lq_bonds!(Complex64);
        checked_full_qr_lq_bonds!(Complex64);
    }
}
