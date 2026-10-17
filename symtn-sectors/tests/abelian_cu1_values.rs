//! The abelian and CU(1) providers keep every value they exposed before
//! their symbols moved to fusion-symbols (#2278), except that an
//! inadmissible F or R is now `0.0`, as in TensorKitSectors.
//!
//! The fixtures are fusion-symbols v0.4.0's `tests/fixtures/` copies, printed
//! from symtn's own `symtn-sectors/src/{abelian,cu1}.rs` at `49d09dc4` /
//! `2a6c1523` (both files are byte-identical to the base of #2278). They are
//! keyed by semantic labels, so this test also checks the `SectorId` codec
//! and argument order of each adapter, not only fusion-symbols. fusion-symbols
//! cross-checks the same values against measured TensorKitSectors 0.3.9.
//!
//! The checked fusion algebra keeps symtn's error contract: a recorded `err`
//! (U(1) or CU(1) fusion leaving the label range) must still be an `Err`.

use symtn_sectors::{
    CU1FusionRule, CU1Irrep, CheckedFusionAlgebra, FermionParityFusionRule, FusionRule,
    MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, SectorId, U1FusionRule, U1Irrep,
    Z2FusionRule, Z2Irrep, ZNFusionRule,
};

const ABELIAN: &str = include_str!("fixtures/symtn-49d09dc4-abelian-values.txt");
const CU1: &str = include_str!("fixtures/symtn-2a6c1523-cu1-values.txt");

fn bits(s: &str) -> u64 {
    u64::from_str_radix(s, 16).expect("hex bits")
}

trait Rule: FusionRule + CheckedFusionAlgebra + MultiplicityFreeRigidSymbols<Scalar = f64> {}
impl<R: FusionRule + CheckedFusionAlgebra + MultiplicityFreeRigidSymbols<Scalar = f64>> Rule for R {}

/// `N`, with symtn's checked fusion overflow (U(1)/CU(1) sums leaving the
/// label range) read as the exact `0`.
fn n<R: Rule>(rule: &R, a: SectorId, b: SectorId, c: SectorId) -> bool {
    rule.try_nsymbol(a, b, c) == Ok(1)
}

fn admissible_f<R: Rule>(rule: &R, [a, b, c, d, e, f]: [SectorId; 6]) -> bool {
    n(rule, a, b, e) && n(rule, e, c, d) && n(rule, b, c, f) && n(rule, a, f, d)
}

/// Admissible: the recorded bits. Inadmissible: exactly `0.0`.
fn check_symbol(got: f64, admissible: bool, recorded: u64, line: &str) {
    if admissible {
        assert_eq!(got.to_bits(), recorded, "{line}");
    } else {
        assert_eq!(got, 0.0, "{line}");
    }
}

/// Unary and N/fusion lines shared by both fixtures. `last` is the recorded
/// value; returns `false` for a line kind this helper does not own.
fn check_unary<R: Rule>(rule: &R, what: &str, a: SectorId, last: &str, line: &str) -> bool {
    let same = |got: f64| assert_eq!(got.to_bits(), bits(last), "{line}");
    match what {
        "dim" => same(rule.dim_scalar(a)),
        "inv_dim" => same(rule.inv_dim_scalar(a)),
        "sqrt_dim" => same(rule.sqrt_dim_scalar(a)),
        "inv_sqrt_dim" => same(rule.inv_sqrt_dim_scalar(a)),
        "twist" => same(rule.twist_scalar(a)),
        _ => return false,
    }
    true
}

fn abelian_line<R: Rule>(rule: &R, id: impl Fn(i64) -> SectorId, w: &[&str], line: &str) {
    let what = w[1];
    let l = |i: usize| id(w[2 + i].parse().expect("label"));
    let last = *w.last().unwrap();
    if check_unary(rule, what, l(0), last, line) {
        return;
    }
    match what {
        "vacuum" => assert_eq!(rule.vacuum(), l(0), "{line}"),
        "dual" => assert_eq!(rule.try_dual_sector(l(0)), Ok(l(1)), "{line}"),
        "fs" => assert_eq!(
            rule.frobenius_schur_phase_scalar(l(0)).to_bits(),
            bits(last),
            "{line}"
        ),
        "fusion" => {
            let got = rule.try_fusion_channels(l(0), l(1));
            if last == "err" {
                assert!(got.is_err(), "{line}");
            } else {
                let want = id(last.parse().expect("channel"));
                assert_eq!(got.expect(line).as_slice(), &[want], "{line}");
            }
        }
        "N" => {
            let got = rule.try_nsymbol(l(0), l(1), l(2));
            if last == "err" {
                assert!(got.is_err(), "{line}");
            } else {
                assert_eq!(got, Ok(last.parse().unwrap()), "{line}");
            }
        }
        "R" => {
            let (a, b, c) = (l(0), l(1), l(2));
            let admissible = n(rule, a, b, c);
            check_symbol(rule.r_symbol_scalar(a, b, c), admissible, bits(last), line);
        }
        "F" => {
            let labels = [l(0), l(1), l(2), l(3), l(4), l(5)];
            let [a, b, c, d, e, f] = labels;
            check_symbol(
                rule.f_symbol_scalar(a, b, c, d, e, f),
                admissible_f(rule, labels),
                bits(last),
                line,
            );
        }
        other => panic!("unknown quantity {other}"),
    }
}

#[test]
fn abelian_values_are_unchanged_on_admissible_labels() {
    let parity = |p: i64| Z2Irrep::new(p as u8).sector_id();
    let u1 = |q: i64| U1Irrep::new(i32::try_from(q).unwrap()).sector_id();
    let mut lines = 0;
    for line in ABELIAN.lines().filter(|l| !l.starts_with('#')) {
        let w: Vec<&str> = line.split_whitespace().collect();
        match w[0] {
            "z2" => abelian_line(&Z2FusionRule, parity, &w, line),
            "fz2" => abelian_line(&FermionParityFusionRule, parity, &w, line),
            "u1" => abelian_line(&U1FusionRule, u1, &w, line),
            zn => {
                let rule = ZNFusionRule::new(zn[2..].parse().expect("zn<N>")).unwrap();
                abelian_line(&rule, |q| rule.irrep(q).into(), &w, line);
            }
        }
        lines += 1;
    }
    assert_eq!(lines, ABELIAN.lines().count() - 1);
}

fn cu1(s: &str) -> CU1Irrep {
    let (q, s) = s.split_once(':').expect("twice_charge:s");
    match (q.parse::<u32>().unwrap(), s) {
        (0, "0") => CU1Irrep::VACUUM,
        (0, "1") => CU1Irrep::PSEUDOSCALAR,
        (q, "2") => CU1Irrep::from_twice_charge(q),
        other => panic!("not a CU(1) label {other:?}"),
    }
}

fn cu1_id(s: &str) -> SectorId {
    cu1(s).sector_id()
}

#[test]
fn cu1_values_are_unchanged_on_admissible_labels() {
    use std::collections::HashMap;

    let rule = CU1FusionRule;
    // Only non-`+0.0` R and F tuples are recorded; the sweep below checks
    // every tuple over the recorded labels.
    let mut symbols = HashMap::<Vec<SectorId>, u64>::new();
    for line in CU1.lines().filter(|l| !l.starts_with('#')) {
        let w: Vec<&str> = line.split_whitespace().collect();
        let last = *w.last().unwrap();
        if w[0] == "max_twice_charge" {
            assert_eq!(
                last.parse::<u32>().unwrap(),
                symtn_sectors::CU1_MAX_TWICE_CHARGE
            );
            continue;
        }
        let a = cu1_id(w[1]);
        if check_unary(&rule, w[0], a, last, line) {
            continue;
        }
        match w[0] {
            "vacuum" => assert_eq!(rule.vacuum(), a, "{line}"),
            "dual" => assert_eq!(rule.try_dual_sector(a), Ok(cu1_id(w[2])), "{line}"),
            "fs" => assert_eq!(
                rule.frobenius_schur_phase_scalar(a).to_bits(),
                bits(w[2]),
                "{line}"
            ),
            "fusion" => {
                let got = rule.try_fusion_channels(a, cu1_id(w[2]));
                if last == "err" {
                    assert!(got.is_err(), "{line}");
                } else {
                    let want: Vec<SectorId> = last.split(',').map(cu1_id).collect();
                    assert_eq!(got.expect(line).as_slice(), want.as_slice(), "{line}");
                }
            }
            "N" => {
                let got = rule.try_nsymbol(a, cu1_id(w[2]), cu1_id(w[3]));
                if last == "err" {
                    assert!(got.is_err(), "{line}");
                } else {
                    assert_eq!(got, Ok(last.parse().unwrap()), "{line}");
                }
            }
            "R" | "F" => {
                let key = w[1..w.len() - 1].iter().map(|s| cu1_id(s)).collect();
                symbols.insert(key, bits(last));
            }
            other => panic!("unknown quantity {other}"),
        }
    }
    let small: Vec<SectorId> = [CU1Irrep::VACUUM, CU1Irrep::PSEUDOSCALAR]
        .into_iter()
        .chain((1..=5).map(CU1Irrep::from_twice_charge))
        .map(Into::into)
        .collect();
    let mut seen = 0;
    let mut recorded = |key: &[SectorId]| {
        seen += usize::from(symbols.contains_key(key));
        symbols.get(key).copied().unwrap_or(0)
    };
    for &a in &small {
        for &b in &small {
            for &c in &small {
                let admissible = n(&rule, a, b, c);
                let want = recorded(&[a, b, c]);
                check_symbol(rule.r_symbol_scalar(a, b, c), admissible, want, "R");
                for &d in &small {
                    for &e in &small {
                        for &f in &small {
                            let labels = [a, b, c, d, e, f];
                            let want = recorded(&labels);
                            check_symbol(
                                rule.f_symbol_scalar(a, b, c, d, e, f),
                                admissible_f(&rule, labels),
                                want,
                                "F",
                            );
                        }
                    }
                }
            }
        }
    }
    assert_eq!(seen, symbols.len(), "every recorded R/F tuple was swept");
}

/// Every F and R tuple over `ids`: `0.0` exactly when a vertex is
/// inadmissible (TensorKitSectors' value), and `±1` otherwise.
fn assert_off_shell_zero<R: Rule>(rule: &R, ids: &[SectorId]) {
    for &a in ids {
        for &b in ids {
            for &c in ids {
                let r = rule.r_symbol_scalar(a, b, c);
                assert_eq!(r == 0.0, !n(rule, a, b, c), "R {a:?} {b:?} {c:?}");
                for &d in ids {
                    for &e in ids {
                        for &f in ids {
                            let labels = [a, b, c, d, e, f];
                            let value = rule.f_symbol_scalar(a, b, c, d, e, f);
                            assert_eq!(value == 0.0, !admissible_f(rule, labels), "F {labels:?}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn inadmissible_f_and_r_are_zero_at_the_label_boundaries() {
    // What: off-shell F/R are 0, including at the Z_N modulus and U(1)
    // charge extremes the fixtures do not reach.
    for modulus in [3, u32::MAX] {
        let rule = ZNFusionRule::new(modulus).unwrap();
        let ids: Vec<SectorId> = [0, 1, i64::from(u32::MAX) - 1]
            .into_iter()
            .map(|q| rule.irrep(q).into())
            .collect();
        assert_off_shell_zero(&rule, &ids);
    }
    let u1: Vec<SectorId> = [i32::MIN + 1, -1, 0, 1, i32::MAX]
        .into_iter()
        .map(|q| U1Irrep::new(q).sector_id())
        .collect();
    assert_off_shell_zero(&U1FusionRule, &u1);
    let parity = [Z2Irrep::EVEN.sector_id(), Z2Irrep::ODD.sector_id()];
    assert_off_shell_zero(&Z2FusionRule, &parity);
    assert_off_shell_zero(&FermionParityFusionRule, &parity);
}
