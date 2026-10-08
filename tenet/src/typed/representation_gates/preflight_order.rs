//! #1995: inv, pinv, exp and solve report the same first error for the same
//! misuse in both fusion modes, in one order — runtime, rule, operand shape
//! (codomain equality, isomorphism, endomorphism), borrowed-view admission,
//! then representation — on multiplicity-free U(1) and SU(2) and on the
//! checked SU(3) rule (racah). The expected first error of every row is the
//! one TensorKit's precedence implies (`src/tensors/linalg.jl` `inv`, `\`,
//! `exp!`; `src/tensors/diagonal.jl` `D \ t`), written out below rather than
//! derived from either mode.

use super::*;
use num_complex::Complex64;
use tenet_core::{SUNFusionRule, ZNFusionRule};

const RUNTIME: &str = "operands belong to different runtimes";
const RULE: &str = "operands use different fusion rules";
const CODOMAIN: &str =
    "invalid argument: solve requires equal divisor and right-hand-side codomains";
const NOT_ISO_SOLVE: &str = "unsupported tensor contraction scope: solve requires an isomorphic divisor codomain and domain";
const NOT_ISO_INV: &str =
    "unsupported tensor contraction scope: inv requires isomorphic codomain and domain";
const NOT_ENDO: &str =
    "unsupported tensor contraction scope: exp requires an endomorphism (codomain == domain)";

fn unsupported_solve() -> String {
    Error::Unsupported {
        operation: "solve",
        alternative: crate::error::Alternative::Materialize,
    }
    .to_string()
}

/// The row text of a result: `ok`, or the error's message without the
/// per-mode wrapper (the facade error type is per mode by design, #1862 D4).
fn text<T, E: std::fmt::Display>(result: &Result<T, E>) -> String {
    match result {
        Ok(_) => "ok".to_string(),
        Err(error) => error
            .to_string()
            .trim_start_matches("operation error: ")
            .to_string(),
    }
}

/// Runs `f`, returning its value and the adjoint payload materializations it
/// made.
fn materializations<T>(f: impl FnOnce() -> T) -> (T, usize) {
    super::ADJOINT_MATERIALIZATIONS.set(Some(0));
    let value = f();
    (
        value,
        super::ADJOINT_MATERIALIZATIONS.replace(None).unwrap(),
    )
}

fn spectrum<R, D>(
    bond: &GradedSpace<R>,
    value: impl Fn(usize) -> D,
) -> Vec<SectorSpectrum<R::Sector, D>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
{
    let mut next = 0;
    bond.sectors()
        .unwrap_or_else(|_| panic!("bond sectors"))
        .into_iter()
        .map(|sector| {
            let degeneracy = bond
                .degeneracy(&sector)
                .unwrap_or_else(|_| panic!("bond degeneracy"));
            let values = (0..degeneracy)
                .map(|_| {
                    next += 1;
                    value(next)
                })
                .collect();
            SectorSpectrum { sector, values }
        })
        .collect()
}

/// The misuse matrix of one rule: `v` and `w` are non-isomorphic legs.
macro_rules! misuse {
    ($v:expr, $w:expr, $dtype:ty, $value:expr) => {{
        let (v, w) = (&$v, &$w);
        let value: fn(f64) -> $dtype = $value;
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let other = Runtime::builder().dense_threads(1).build().unwrap();
        let rand = |rt: &Runtime, cod: &GradedSpace<_>, dom: &GradedSpace<_>, seed| {
            TensorMap::<_, $dtype>::rand_with_seed(rt, [cod], [dom], seed).unwrap()
        };
        let vv = rand(&runtime, v, v, 1);
        let vw = rand(&runtime, v, w, 2);
        let wv = rand(&runtime, w, v, 3);
        let wv_other = rand(&other, w, v, 4);
        let vw_other = rand(&other, v, w, 5);
        let y = rand(&runtime, v, v, 6);
        let d: TensorMap<_, $dtype> =
            TensorMap::diagonal(&runtime, v, spectrum(v, |k| value(1.0 + 0.5 * k as f64))).unwrap();
        let singular: TensorMap<_, $dtype> = TensorMap::diagonal(
            &runtime,
            v,
            spectrum(v, |k| {
                <$dtype>::from_real(if k == 2 { 0.0 } else { k as f64 })
            }),
        )
        .unwrap();
        let solve = |lhs: &TensorMap<_, $dtype>, rhs: TensorRef<'_, _, $dtype>| {
            text(&lhs.solve(&[0], &[1], rhs, &[0], &[1]))
        };
        let lazy_wv = wv.adjoint().unwrap();
        let (lazy_exp, lazy_exp_copies) = materializations(|| lazy_wv.exp(&[0], &[1]));

        // `D \ y'` reads the view: equal to the owned adjoint, and to the
        // dense LU route on the materialized divisor (an independent oracle).
        let (compact_view, copies) =
            materializations(|| d.solve(&[0], &[1], y.adjoint_view(), &[0], &[1]));
        let y_owned = y.adjoint().unwrap().materialize().unwrap();
        let compact_view_ok = compact_view.as_ref().map(|x| {
            let lu = d
                .materialize()
                .unwrap()
                .solve(&[0], &[1], &y_owned, &[0], &[1])
                .unwrap();
            let owned = d.solve(&[0], &[1], &y_owned, &[0], &[1]).unwrap();
            assert_eq!(x.dense_data().unwrap(), owned.dense_data().unwrap());
            let diff = x
                .axpby(<$dtype>::from_real(1.0), &lu, <$dtype>::from_real(-1.0))
                .unwrap()
                .norm(2.0)
                .unwrap();
            assert!(diff < 1e-12 * lu.norm(2.0).unwrap().max(1.0), "{diff}");
            copies
        });

        vec![
            ("solve runtime", solve(&vw, (&wv_other).into())),
            (
                "solve runtime borrowed",
                solve(&vw, vw_other.adjoint_view()),
            ),
            ("solve codomain", solve(&vv, (&wv).into())),
            ("solve codomain borrowed", solve(&vv, vw.adjoint_view())),
            ("solve non-isomorphic", solve(&vw, (&vv).into())),
            (
                "solve non-isomorphic borrowed",
                solve(&vw, y.adjoint_view()),
            ),
            ("solve borrowed dense divisor", solve(&vv, y.adjoint_view())),
            (
                "solve borrowed compact divisor",
                match compact_view_ok {
                    Ok(copies) => format!("ok, {copies} copies"),
                    Err(error) => error.to_string(),
                },
            ),
            (
                "solve singular compact divisor",
                solve(&singular, (&y).into()),
            ),
            (
                "solve singular compact divisor borrowed",
                solve(&singular, y.adjoint_view()),
            ),
            ("inv non-isomorphic", text(&vw.inv(&[0], &[1]))),
            ("inv non-isomorphic lazy", text(&lazy_wv.inv(&[0], &[1]))),
            ("exp non-endomorphic", text(&vw.exp(&[0], &[1]))),
            (
                "exp non-endomorphic lazy",
                format!("{}, {lazy_exp_copies} copies", text(&lazy_exp)),
            ),
            ("pinv rcond", text(&vw.pinv(&[0], &[1], -1.0))),
        ]
    }};
}

fn expected_misuse(singular: &str) -> Vec<(&'static str, String)> {
    vec![
        ("solve runtime", RUNTIME.to_string()),
        ("solve runtime borrowed", RUNTIME.to_string()),
        ("solve codomain", CODOMAIN.to_string()),
        ("solve codomain borrowed", CODOMAIN.to_string()),
        ("solve non-isomorphic", NOT_ISO_SOLVE.to_string()),
        ("solve non-isomorphic borrowed", NOT_ISO_SOLVE.to_string()),
        ("solve borrowed dense divisor", unsupported_solve()),
        ("solve borrowed compact divisor", "ok, 0 copies".to_string()),
        ("solve singular compact divisor", singular.to_string()),
        (
            "solve singular compact divisor borrowed",
            singular.to_string(),
        ),
        ("inv non-isomorphic", NOT_ISO_INV.to_string()),
        ("inv non-isomorphic lazy", NOT_ISO_INV.to_string()),
        ("exp non-endomorphic", NOT_ENDO.to_string()),
        ("exp non-endomorphic lazy", format!("{NOT_ENDO}, 0 copies")),
        (
            "pinv rcond",
            "invalid argument: pinv rcond must be finite and non-negative".to_string(),
        ),
    ]
}

fn differing(name: &str, actual: &[(&str, String)], expected: &[(&str, String)]) -> Vec<String> {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .filter(|((_, actual), (_, expected))| actual != expected)
        .map(|((op, actual), (_, expected))| {
            format!("{name}: {op}: got `{actual}`, want `{expected}`")
        })
        .collect()
}

#[test]
fn misuse_reports_one_first_error_in_both_modes() {
    let u1 = Arc::new(U1FusionRule);
    let u1_leg = |pairs: &[(i32, usize)]| {
        GradedSpace::try_new(
            Arc::clone(&u1),
            pairs.iter().map(|&(q, d)| (U1Irrep::new(q), d)),
        )
        .unwrap()
    };
    let su2 = Arc::new(SU2FusionRule);
    let su2_leg = |pairs: &[(usize, usize)]| {
        GradedSpace::try_new(
            Arc::clone(&su2),
            pairs
                .iter()
                .map(|&(j, d)| (SU2Irrep::from_twice_spin(j), d)),
        )
        .unwrap()
    };
    let su3 = Arc::new(SUNFusionRule::new(3).unwrap());
    let su3_leg = |pairs: &[(&[i64], usize)]| {
        GradedSpace::try_new(
            Arc::clone(&su3),
            pairs.iter().map(|&(l, d)| (l.to_vec(), d)),
        )
        .unwrap()
    };
    let real = |x: f64| x;
    let complex = |x: f64| Complex64::new(x, 0.5 - x);

    // The singular-divisor error is the dense route's own, in every mode;
    // take its text from one row and require every rule and dtype to agree.
    let reference = misuse!(u1_leg(&[(0, 2), (1, 3)]), u1_leg(&[(0, 2)]), f64, real);
    let singular = reference
        .iter()
        .find(|(op, _)| *op == "solve singular compact divisor")
        .unwrap()
        .1
        .clone();
    assert_ne!(singular, "ok");
    let expected = expected_misuse(&singular);
    let mut differing_rows = differing("U1 f64", &reference, &expected);
    let mut assert_rows = |name: &str, actual: &[(&str, String)], expected: &[(&str, String)]| {
        differing_rows.extend(differing(name, actual, expected));
    };
    assert_rows(
        "U1 c64",
        &misuse!(
            u1_leg(&[(0, 2), (1, 3)]),
            u1_leg(&[(0, 2)]),
            Complex64,
            complex
        ),
        &expected,
    );
    assert_rows(
        "SU2 f64",
        &misuse!(su2_leg(&[(0, 2), (1, 3)]), su2_leg(&[(0, 2)]), f64, real),
        &expected,
    );
    assert_rows(
        "SU2 c64",
        &misuse!(
            su2_leg(&[(0, 2), (1, 3)]),
            su2_leg(&[(0, 2)]),
            Complex64,
            complex
        ),
        &expected,
    );
    assert_rows(
        "SU3 checked f64",
        &misuse!(
            su3_leg(&[(&[0, 0], 2), (&[1, 0], 3)]),
            su3_leg(&[(&[0, 0], 2)]),
            f64,
            real
        ),
        &expected,
    );
    assert_rows(
        "SU3 checked c64",
        &misuse!(
            su3_leg(&[(&[0, 0], 2), (&[1, 0], 3)]),
            su3_leg(&[(&[0, 0], 2)]),
            Complex64,
            complex
        ),
        &expected,
    );
    assert!(differing_rows.is_empty(), "{}", differing_rows.join("\n"));
}

/// A rule mismatch precedes every later check, including a borrowed rhs and
/// one whose moved roles would copy it.
macro_rules! rule_rows {
    ($divisor_leg:expr, $rhs_leg:expr, $dtype:ty) => {{
        let (a, b) = (&$divisor_leg, &$rhs_leg);
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let divisor = TensorMap::<_, $dtype>::rand_with_seed(&runtime, [a], [a], 1).unwrap();
        let rhs = TensorMap::<_, $dtype>::rand_with_seed(&runtime, [b], [b], 2).unwrap();
        vec![
            text(&divisor.solve(&[0], &[1], &rhs, &[0], &[1])),
            text(&divisor.solve(&[0], &[1], rhs.adjoint_view(), &[0], &[1])),
            text(&divisor.solve(&[1], &[0], rhs.adjoint_view(), &[1], &[0])),
        ]
    }};
}

#[test]
fn rule_mismatch_is_the_first_error_in_both_modes() {
    let z2 = Arc::new(ZNFusionRule::new(2).unwrap());
    let z3 = Arc::new(ZNFusionRule::new(3).unwrap());
    let za = GradedSpace::try_new(Arc::clone(&z2), [(z2.irrep(0), 2), (z2.irrep(1), 1)]).unwrap();
    let zb = GradedSpace::try_new(Arc::clone(&z3), [(z3.irrep(0), 1)]).unwrap();
    let su3 = Arc::new(SUNFusionRule::new(3).unwrap());
    let su4 = Arc::new(SUNFusionRule::new(4).unwrap());
    let sa = GradedSpace::try_new(Arc::clone(&su3), [(vec![0, 0], 2), (vec![1, 0], 1)]).unwrap();
    let sb = GradedSpace::try_new(Arc::clone(&su4), [(vec![0, 0, 0], 1)]).unwrap();
    let expected = vec![RULE.to_string(); 3];
    assert_eq!(rule_rows!(za, zb, f64), expected, "Z2/Z3 f64");
    assert_eq!(rule_rows!(za, zb, Complex64), expected, "Z2/Z3 c64");
    assert_eq!(rule_rows!(sa, sb, f64), expected, "SU3/SU4 f64");
    assert_eq!(rule_rows!(sa, sb, Complex64), expected, "SU3/SU4 c64");
}
