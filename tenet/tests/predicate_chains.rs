//! #1557 removed `is_hermitian`, `is_antihermitian`, `is_isometric`,
//! `is_unitary`, `is_posdef`, `project_hermitian` and `project_antihermitian`:
//! each was a short chain of public operations. These tests pin the chains in
//! `common/predicate_chains.rs` to the answers the removed methods gave at
//! 388d24a9: every truth-table entry below and every projection bit pattern
//! was first asserted equal to the removed method's output on these fixtures.

#[path = "../../tests/support/numerics.rs"]
mod numerics;

include!("common/predicate_chains.rs");
include!("common/predicate_chain_coefficients.rs");

use num_complex::{Complex32, Complex64};
use numerics::Numeric;
use tenet::core::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use tenet::prelude::{GradedSpace, Runtime, TensorMap};
use tenet::typed::{Eigh, LeftPolar, Qr};

/// Projection payloads `(re, im)` as `f64` bits, captured from the removed
/// `project_hermitian` / `project_antihermitian` at 388d24a9 on these fixtures.
const U1_F64_HERMITIAN: [(u64, u64); 13] = [
    (0xbfe39456ffecc09d, 0x0000000000000000),
    (0xbfe41e07726001b8, 0x0000000000000000),
    (0xbfe41e07726001b8, 0x0000000000000000),
    (0x3feb36c6dc1d7445, 0x0000000000000000),
    (0xbfe39456ffecc09d, 0x0000000000000000),
    (0xbfe41e07726001b8, 0x0000000000000000),
    (0x3fe4d07c1c0eaa1f, 0x0000000000000000),
    (0xbfe41e07726001b8, 0x0000000000000000),
    (0x3feb36c6dc1d7445, 0x0000000000000000),
    (0x3fd602011470c412, 0x0000000000000000),
    (0x3fe4d07c1c0eaa1f, 0x0000000000000000),
    (0x3fd602011470c412, 0x0000000000000000),
    (0xbfef63d460731b8d, 0x0000000000000000),
];
const U1_F64_ANTIHERMITIAN: [(u64, u64); 13] = [
    (0x0000000000000000, 0x0000000000000000),
    (0xbfd04e7680b44646, 0x0000000000000000),
    (0x3fd04e7680b44646, 0x0000000000000000),
    (0x0000000000000000, 0x0000000000000000),
    (0x0000000000000000, 0x0000000000000000),
    (0xbfd04e7680b44646, 0x0000000000000000),
    (0xbfd5b0ad08344aba, 0x0000000000000000),
    (0x3fd04e7680b44646, 0x0000000000000000),
    (0x0000000000000000, 0x0000000000000000),
    (0x3fd46c1569ec7ff4, 0x0000000000000000),
    (0x3fd5b0ad08344aba, 0x0000000000000000),
    (0xbfd46c1569ec7ff4, 0x0000000000000000),
    (0x0000000000000000, 0x0000000000000000),
];
const U1_C64_HERMITIAN: [(u64, u64); 13] = [
    (0xbfe39456ffecc09d, 0x0000000000000000),
    (0xbfe41e07726001b8, 0xbfd13ea14428ef84),
    (0xbfe41e07726001b8, 0x3fd13ea14428ef84),
    (0x3feb36c6dc1d7445, 0x0000000000000000),
    (0xbfe39456ffecc09d, 0x0000000000000000),
    (0xbfe41e07726001b8, 0xbfd13ea14428ef84),
    (0x3fe4d07c1c0eaa1f, 0x3fdedddccae6ad6d),
    (0xbfe41e07726001b8, 0x3fd13ea14428ef84),
    (0x3feb36c6dc1d7445, 0x0000000000000000),
    (0x3fd602011470c412, 0x3fc6c6806e3a6972),
    (0x3fe4d07c1c0eaa1f, 0xbfdedddccae6ad6d),
    (0x3fd602011470c412, 0xbfc6c6806e3a6972),
    (0xbfef63d460731b8d, 0x0000000000000000),
];
const U1_C64_ANTIHERMITIAN: [(u64, u64); 13] = [
    (0x0000000000000000, 0xbfec4542b2ba24db),
    (0xbfd04e7680b44646, 0x3fe297763a08fc83),
    (0x3fd04e7680b44646, 0x3fe297763a08fc83),
    (0x0000000000000000, 0x3fe5370b3f2ea203),
    (0x0000000000000000, 0xbfec4542b2ba24db),
    (0xbfd04e7680b44646, 0x3fe297763a08fc83),
    (0xbfd5b0ad08344aba, 0x3fe039e43ab578c6),
    (0x3fd04e7680b44646, 0x3fe297763a08fc83),
    (0x0000000000000000, 0x3fe5370b3f2ea203),
    (0x3fd46c1569ec7ff4, 0xbfe9b23444e48130),
    (0x3fd5b0ad08344aba, 0x3fe039e43ab578c6),
    (0xbfd46c1569ec7ff4, 0xbfe9b23444e48130),
    (0x0000000000000000, 0xbfd6ed3a82eb4f98),
];
const SU2_F64_HERMITIAN: [(u64, u64); 13] = [
    (0xbfe39456ffecc09d, 0x0000000000000000),
    (0xbfe41e07726001b8, 0x0000000000000000),
    (0x3fe4d07c1c0eaa1f, 0x0000000000000000),
    (0xbfe41e07726001b8, 0x0000000000000000),
    (0x3feb36c6dc1d7445, 0x0000000000000000),
    (0x3fd602011470c412, 0x0000000000000000),
    (0x3fe4d07c1c0eaa1f, 0x0000000000000000),
    (0x3fd602011470c412, 0x0000000000000000),
    (0xbfef63d460731b8d, 0x0000000000000000),
    (0xbfe39456ffecc09d, 0x0000000000000000),
    (0xbfe41e07726001b8, 0x0000000000000000),
    (0xbfe41e07726001b8, 0x0000000000000000),
    (0x3feb36c6dc1d7445, 0x0000000000000000),
];
const SU2_F64_ANTIHERMITIAN: [(u64, u64); 13] = [
    (0x0000000000000000, 0x0000000000000000),
    (0xbfd04e7680b44646, 0x0000000000000000),
    (0xbfd5b0ad08344aba, 0x0000000000000000),
    (0x3fd04e7680b44646, 0x0000000000000000),
    (0x0000000000000000, 0x0000000000000000),
    (0x3fd46c1569ec7ff4, 0x0000000000000000),
    (0x3fd5b0ad08344aba, 0x0000000000000000),
    (0xbfd46c1569ec7ff4, 0x0000000000000000),
    (0x0000000000000000, 0x0000000000000000),
    (0x0000000000000000, 0x0000000000000000),
    (0xbfd04e7680b44646, 0x0000000000000000),
    (0x3fd04e7680b44646, 0x0000000000000000),
    (0x0000000000000000, 0x0000000000000000),
];
const SU2_C32_HERMITIAN: [(u64, u64); 13] = [
    (0xbfe3945700000000, 0x0000000000000000),
    (0xbfe41e0780000000, 0xbfd13ea140000000),
    (0x3fe4d07c20000000, 0x3fdedddcc0000000),
    (0xbfe41e0780000000, 0x3fd13ea140000000),
    (0x3feb36c6e0000000, 0x0000000000000000),
    (0x3fd6020120000000, 0x3fc6c68080000000),
    (0x3fe4d07c20000000, 0xbfdedddcc0000000),
    (0x3fd6020120000000, 0xbfc6c68080000000),
    (0xbfef63d460000000, 0x0000000000000000),
    (0xbfe3945700000000, 0x0000000000000000),
    (0xbfe41e0780000000, 0xbfd13ea140000000),
    (0xbfe41e0780000000, 0x3fd13ea140000000),
    (0x3feb36c6e0000000, 0x0000000000000000),
];
const SU2_C32_ANTIHERMITIAN: [(u64, u64); 13] = [
    (0x0000000000000000, 0xbfec4542c0000000),
    (0xbfd04e7680000000, 0x3fe2977640000000),
    (0xbfd5b0ad00000000, 0x3fe039e440000000),
    (0x3fd04e7680000000, 0x3fe2977640000000),
    (0x0000000000000000, 0x3fe5370b40000000),
    (0x3fd46c1560000000, 0xbfe9b23440000000),
    (0x3fd5b0ad00000000, 0x3fe039e440000000),
    (0xbfd46c1560000000, 0xbfe9b23440000000),
    (0x0000000000000000, 0xbfd6ed3a80000000),
    (0x0000000000000000, 0xbfec4542c0000000),
    (0xbfd04e7680000000, 0x3fe2977640000000),
    (0x3fd04e7680000000, 0x3fe2977640000000),
    (0x0000000000000000, 0x3fe5370b40000000),
];

fn bits<D: Numeric>(data: &[D]) -> Vec<(u64, u64)> {
    data.iter()
        .map(|value| {
            let wide = value.wide();
            (wide.re.to_bits(), wide.im.to_bits())
        })
        .collect()
}

fn fill(ij: &[usize]) -> f64 {
    let key = ij
        .iter()
        .enumerate()
        .map(|(axis, &index)| (axis + 2) * (index + 1))
        .sum::<usize>();
    (key as f64 * 0.7 + 0.3).sin()
}

/// One multiplicity-free family: the truth table of every predicate, old
/// against chain, and the projection payloads, returned as bits.
macro_rules! pin_family {
    ($rt:expr, $leg:expr, $d:ty, $tol:expr, $value:expr, $hermitian:expr, $antihermitian:expr) => {{
        let rt: &Runtime = $rt;
        let leg = $leg;
        let value = $value;
        let x: TensorMap<_, $d> =
            TensorMap::from_block_fn(rt, [&leg], [&leg], |_, ij: &[usize]| value(ij)).unwrap();
        let h = project_hermitian!(x).unwrap();
        let a = project_antihermitian!(x).unwrap();
        let identity = TensorMap::id(rt, [&leg]).unwrap();
        let positive = x
            .adjoint()
            .unwrap()
            .compose(&x)
            .unwrap()
            .axpby(
                ChainCoefficient::real(1.0),
                &identity,
                ChainCoefficient::real(1.0),
            )
            .unwrap();
        let negative = positive.scale(ChainCoefficient::real(-1.0));
        // A polar factor, not a QR `q`: a 2x2 Householder `q` is a symmetric
        // reflector, which would also be Hermitian.
        let LeftPolar { w: unitary, .. } = x.left_polar().unwrap();
        let tall: TensorMap<_, $d> =
            TensorMap::from_block_fn(rt, [&leg, &leg], [&leg], |_, ij: &[usize]| value(ij))
                .unwrap();
        let Qr { q: isometry, .. } = tall.qr_compact().unwrap();
        let Eigh {
            d: compact_positive,
            ..
        } = positive.eigh_full().unwrap();
        let Eigh {
            d: compact_negative,
            ..
        } = negative.eigh_full().unwrap();

        let tol: f64 = $tol;
        let dense = [
            ("x", &x, [false, false, false, false, false]),
            ("hermitian", &h, [true, false, false, false, false]),
            ("antihermitian", &a, [false, true, false, false, false]),
            ("positive", &positive, [true, false, false, false, true]),
            ("negative", &negative, [true, false, false, false, false]),
            ("unitary", &unitary, [false, false, true, true, false]),
            ("isometry", &isometry, [false, false, true, false, false]),
        ];
        for (name, t, expected) in dense {
            let chain = [
                is_hermitian!(t, tol),
                is_antihermitian!(t, tol),
                is_isometric!(t, tol),
                is_unitary!(t, tol),
                is_posdef!(t, tol),
            ];
            assert_eq!(chain, expected, "{name}");
        }
        let re = |value: $d| value.wide().re;
        for (name, t, expected) in [
            ("compact positive", &compact_positive, true),
            ("compact negative", &compact_negative, false),
        ] {
            assert_eq!(is_posdef_compact!(t, tol, re), expected, "{name}");
        }
        assert!(project_hermitian!(isometry).is_err());
        assert!(project_antihermitian!(isometry).is_err());

        assert_eq!(bits(h.data()), $hermitian);
        assert_eq!(bits(a.data()), $antihermitian);
    }};
}

fn u1_leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(U1FusionRule, [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)]).unwrap()
}

fn su2_leg() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        SU2FusionRule,
        [
            (SU2Irrep::from_twice_spin(0), 3),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap()
}

#[test]
fn u1_chains_match_the_removed_methods() {
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    pin_family!(
        &rt,
        u1_leg(),
        f64,
        1e-10,
        fill,
        U1_F64_HERMITIAN,
        U1_F64_ANTIHERMITIAN
    );
    pin_family!(
        &rt,
        u1_leg(),
        Complex64,
        1e-10,
        |ij: &[usize]| Complex64::new(fill(ij), fill(&[ij[0] + 1, ij[1]])),
        U1_C64_HERMITIAN,
        U1_C64_ANTIHERMITIAN
    );
}

#[test]
fn su2_chains_match_the_removed_methods() {
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    pin_family!(
        &rt,
        su2_leg(),
        f64,
        1e-10,
        fill,
        SU2_F64_HERMITIAN,
        SU2_F64_ANTIHERMITIAN
    );
    pin_family!(
        &rt,
        su2_leg(),
        Complex32,
        1e-4,
        |ij: &[usize]| Complex32::new(fill(ij) as f32, fill(&[ij[0] + 1, ij[1]]) as f32),
        SU2_C32_HERMITIAN,
        SU2_C32_ANTIHERMITIAN
    );
}

/// Checked-Generic providers never had the removed methods; the chains are
/// the only spelling there. SU(3) with an adjoint leg carries outer
/// multiplicity. Two capability differences from the multiplicity-free
/// chains: `TensorMap::id` is multiplicity-free-only, so the identity is
/// `powi(0)`; and checked-Generic `compose` rejects a lazy adjoint operand, so
/// `t†` enters the Gram map materialized.
#[cfg(feature = "racah-generated")]
#[test]
fn checked_generic_su3_chains_decide_true_and_false_cases() {
    use std::cell::Cell;
    use std::sync::Arc;

    use tenet::typed::SUNFusionRule;

    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(vec![1, 1], 2)]).unwrap();
    let materialized_adjoint = |t: &TensorMap<SUNFusionRule, f64>| {
        let logical = t.adjoint().unwrap().data().to_vec();
        let position = Cell::new(0usize);
        TensorMap::from_block_fn(&rt, t.domain().iter(), t.codomain().iter(), |_, _| {
            let index = position.get();
            position.set(index + 1);
            logical[index]
        })
        .unwrap()
    };
    let is_identity = |gram: TensorMap<SUNFusionRule, f64>, tol: f64| {
        let identity = gram.powi(0).unwrap();
        gram.axpby(1.0, &identity, -1.0).unwrap().norm(2.0).unwrap()
            <= tol * gram.norm(2.0).unwrap().max(1.0)
    };
    // `t† ∘ t == id`, and for unitarity also `t ∘ t† == id`.
    let is_isometric = |t: &TensorMap<SUNFusionRule, f64>, tol: f64| {
        is_identity(materialized_adjoint(t).compose(t).unwrap(), tol)
    };
    let is_unitary = |t: &TensorMap<SUNFusionRule, f64>, tol: f64| {
        is_isometric(t, tol) && is_identity(t.compose(&materialized_adjoint(t)).unwrap(), tol)
    };
    let x: TensorMap<_, f64> =
        TensorMap::from_block_fn(&rt, [&leg, &leg], [&leg, &leg], |_, ij: &[usize]| fill(ij))
            .unwrap();
    let h = project_hermitian!(x).unwrap();
    let a = project_antihermitian!(x).unwrap();
    let positive = h
        .compose(&h)
        .unwrap()
        .axpby(1.0, &x.powi(0).unwrap(), 1.0)
        .unwrap();
    let negative = positive.scale(-1.0);
    let Qr { q: unitary, .. } = x.qr_compact().unwrap();
    let tall: TensorMap<_, f64> =
        TensorMap::from_block_fn(&rt, [&leg, &leg], [&leg], |_, ij: &[usize]| fill(ij)).unwrap();
    let Qr { q: isometry, .. } = tall.qr_compact().unwrap();
    assert!(project_hermitian!(isometry).is_err());

    let tol = 1e-10;
    for (name, t, expected) in [
        ("x", &x, [false, false, false, false, false]),
        ("hermitian", &h, [true, false, false, false, false]),
        ("antihermitian", &a, [false, true, false, false, false]),
        ("positive", &positive, [true, false, false, false, true]),
        ("negative", &negative, [true, false, false, false, false]),
        ("isometry", &isometry, [false, false, true, false, false]),
    ] {
        let chain = [
            is_hermitian!(t, tol),
            is_antihermitian!(t, tol),
            is_isometric(t, tol),
            is_unitary(t, tol),
            is_posdef!(t, tol),
        ];
        assert_eq!(chain, expected, "{name}");
    }
    // `q` of a square QR is unitary; whether it is also Hermitian depends on
    // the reflectors, so only unitarity is asserted.
    assert!(is_unitary(&unitary, tol));
}
