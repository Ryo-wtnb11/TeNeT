//! Reference tests for the CategoryData Fibonacci provider (#539 TODO 3).
//!
//! Three independent kinds of check, deliberately kept apart:
//!
//! 1. **Oracle comparison against the pinned Julia environment.** Every `N`,
//!    `F` and `R` entry and every derived value is compared with
//!    `fixtures/fib-categorydata-v0.1.3.txt`, emitted by CategoryData v0.3.6 /
//!    TensorKitSectors v0.3.9 (see `fixtures/generate_fixture.jl`). The fixture
//!    goes through CategoryData.jl's own parser, indexing and derivation, so
//!    it exercises this crate's import path rather than echoing the same file.
//! 2. **Oracle comparison against the closed-form provider.** The two are
//!    independent implementations of one category and must agree to within a
//!    stated floating-point bound, while carrying different `RuleIdentity`s.
//! 3. **Consistency checks.** Unitarity, triangle, pentagon, hexagon and the
//!    ribbon braid relation. These catch import, indexing, multiplicity-axis
//!    and gauge mistakes. They are *never* generators: nothing here solves a
//!    coherence equation, and nothing derives `F` or `R` from `N`.

use std::collections::HashMap;

use num_complex::Complex64;
use tenet_category_data::{CategoryDataFibonacci, CategoryObject};
use tenet_sectors::{
    CheckedFusionAlgebra, FibonacciFusionRule, FusionAlgebraError, FusionRule,
    MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, SectorCodec, SectorId,
};

const FIXTURE: &str = include_str!("fixtures/fib-categorydata-v0.1.3.txt");

/// Objects of the shipped category, as zero-based ids.
const SECTORS: [SectorId; 2] = [SectorId::new(0), SectorId::new(1)];

/// Documented floating-point bound for every *approximate* comparison in this
/// file: the table-vs-closed-form agreement and the coherence residuals.
///
/// `32 * f64::EPSILON` is a fixed multiple of the unit round-off, not a tuned
/// constant. The largest expression compared is the pentagon equation, whose
/// two sides are sums of at most `rank = 2` triple products of unit-modulus
/// coefficients; a few dozen ulps covers the accumulated rounding of that with
/// a large margin, and no term here has a magnitude far from one, so an
/// absolute bound is the honest form. It is deliberately *not* per-category
/// tunable — a category that needs its own tolerance has an import bug.
const FP_BOUND: f64 = 32.0 * f64::EPSILON;

fn table() -> CategoryDataFibonacci {
    CategoryDataFibonacci::try_new().expect("the shipped tables load")
}

fn assert_bits(actual: Complex64, expected: Complex64, what: &str) {
    assert_eq!(
        (actual.re.to_bits(), actual.im.to_bits()),
        (expected.re.to_bits(), expected.im.to_bits()),
        "{what}: {actual} != {expected}"
    );
}

/// Signed distance in representable `f64`s, the standard ULP measure.
fn ulps(a: f64, b: f64) -> i64 {
    let key = |x: f64| {
        let bits = x.to_bits() as i64;
        if bits < 0 {
            i64::MIN - bits
        } else {
            bits
        }
    };
    (key(a) - key(b)).abs()
}

fn complex_ulps(a: Complex64, b: Complex64) -> i64 {
    ulps(a.re, b.re).max(ulps(a.im, b.im))
}

// -------------------------------------------------------------------------
// 1. The pinned CategoryData.jl / TensorKitSectors fixture
// -------------------------------------------------------------------------

/// `tag` and one-based labels -> value, as emitted by the Julia generator.
fn fixture() -> HashMap<(String, Vec<usize>), Complex64> {
    let mut entries = HashMap::new();
    for line in FIXTURE.lines() {
        if line.trim_start().starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        let tag = fields[0].to_owned();
        // `dual` and `N` carry one trailing integer; everything else carries a
        // `re im` pair.
        let (labels, value) = if matches!(tag.as_str(), "dual" | "N") {
            let (head, tail) = fields[1..].split_at(fields.len() - 2);
            (head, Complex64::new(tail[0].parse().unwrap(), 0.0))
        } else {
            let (head, tail) = fields[1..].split_at(fields.len() - 3);
            (
                head,
                Complex64::new(tail[0].parse().unwrap(), tail[1].parse().unwrap()),
            )
        };
        let key = (tag, labels.iter().map(|f| f.parse().unwrap()).collect());
        assert!(
            entries.insert(key, value).is_none(),
            "duplicate fixture row"
        );
    }
    entries
}

fn fixture_value(
    entries: &HashMap<(String, Vec<usize>), Complex64>,
    tag: &str,
    labels: &[usize],
) -> Complex64 {
    // Labels are one-based in the fixture, matching the source files.
    let key = (
        tag.to_owned(),
        labels.iter().map(|&id| id + 1).collect::<Vec<_>>(),
    );
    *entries
        .get(&key)
        .unwrap_or_else(|| panic!("fixture is missing {tag}{labels:?}"))
}

#[test]
fn every_fusion_and_symbol_entry_matches_the_pinned_julia_environment() {
    let fib = table();
    let entries = fixture();
    let mut checked = 0usize;

    for a in SECTORS {
        assert_eq!(
            fib.dual(a).id(),
            fixture_value(&entries, "dual", &[a.id()]).re as usize - 1,
            "dual of {a:?}"
        );
        checked += 1;
        for b in SECTORS {
            for c in SECTORS {
                assert_eq!(
                    fib.nsymbol(a, b, c) as f64,
                    fixture_value(&entries, "N", &[a.id(), b.id(), c.id()]).re,
                    "N{:?}",
                    (a, b, c)
                );
                assert_bits(
                    fib.r_symbol_scalar(a, b, c),
                    fixture_value(&entries, "R", &[a.id(), b.id(), c.id()]),
                    "R",
                );
                checked += 2;
                for d in SECTORS {
                    for e in SECTORS {
                        for f in SECTORS {
                            assert_bits(
                                fib.f_symbol_scalar(a, b, c, d, e, f),
                                fixture_value(
                                    &entries,
                                    "F",
                                    &[a.id(), b.id(), c.id(), d.id(), e.id(), f.id()],
                                ),
                                "F",
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
    }

    // 2 duals + 8 N + 8 R + 64 F: the full rank-2 tables, nothing sampled.
    assert_eq!(checked, 2 + 8 + 8 + 64);
}

#[test]
fn every_derived_value_matches_the_pinned_julia_environment() {
    let fib = table();
    let entries = fixture();

    let mut checked = 0usize;

    for a in SECTORS {
        assert_bits(
            fib.dim_scalar(a),
            fixture_value(&entries, "dim", &[a.id()]),
            "dim",
        );
        assert_bits(
            fib.frobenius_schur_phase_scalar(a),
            fixture_value(&entries, "frobeniusschur", &[a.id()]),
            "frobenius-schur",
        );
        assert_bits(
            fib.twist_scalar(a),
            fixture_value(&entries, "twist", &[a.id()]),
            "twist",
        );
        checked += 3;
    }

    // A and B come from the `tenet-sectors` trait defaults, so this also pins
    // that TeNeT composes them exactly as TensorKitSectors does.
    for a in SECTORS {
        for b in SECTORS {
            for c in SECTORS {
                let labels = [a.id(), b.id(), c.id()];
                assert_bits(
                    fib.a_symbol_scalar(a, b, c),
                    fixture_value(&entries, "A", &labels),
                    "A",
                );
                assert_bits(
                    fib.b_symbol_scalar(a, b, c),
                    fixture_value(&entries, "B", &labels),
                    "B",
                );
                checked += 2;
            }
        }
    }

    // 2 x (dim + Frobenius-Schur + twist) + 8 A + 8 B.
    assert_eq!(checked, 6 + 8 + 8);
}

// -------------------------------------------------------------------------
// 2. The closed-form provider as an independent oracle
// -------------------------------------------------------------------------

#[test]
fn table_and_closed_form_agree_within_the_documented_bound() {
    let fib = table();
    let closed = FibonacciFusionRule;
    let mut worst_ulps = 0i64;
    let mut worst_where = String::new();
    let mut record = |actual: Complex64, expected: Complex64, what: String| {
        let distance = (actual - expected).norm();
        assert!(
            distance <= FP_BOUND,
            "{what}: {actual} vs {expected} ({distance:e})"
        );
        if complex_ulps(actual, expected) > worst_ulps {
            worst_ulps = complex_ulps(actual, expected);
            worst_where = what;
        }
    };

    for a in SECTORS {
        // Fusion structure is exact integer data and must agree exactly.
        assert_eq!(fib.dual(a), closed.dual(a));
        record(fib.dim_scalar(a), closed.dim_scalar(a), format!("dim{a:?}"));
        record(
            fib.twist_scalar(a),
            closed.twist_scalar(a),
            format!("twist{a:?}"),
        );
        record(
            fib.frobenius_schur_phase_scalar(a),
            closed.frobenius_schur_phase_scalar(a),
            format!("kappa{a:?}"),
        );
        for b in SECTORS {
            assert_eq!(
                fib.fusion_channels(a, b).as_slice(),
                closed.fusion_channels(a, b).as_slice()
            );
            for c in SECTORS {
                assert_eq!(fib.nsymbol(a, b, c), closed.nsymbol(a, b, c));
                record(
                    fib.r_symbol_scalar(a, b, c),
                    closed.r_symbol_scalar(a, b, c),
                    format!("R{:?}", (a, b, c)),
                );
                for d in SECTORS {
                    for e in SECTORS {
                        for f in SECTORS {
                            record(
                                fib.f_symbol_scalar(a, b, c, d, e, f),
                                closed.f_symbol_scalar(a, b, c, d, e, f),
                                format!("F{:?}", (a, b, c, d, e, f)),
                            );
                        }
                    }
                }
            }
        }
    }

    // Reported, not asserted at a specific value: the point of the bound above
    // is that it is a stated property, while the measured ulp gap is an
    // observation about today's two evaluations.
    println!("worst table-vs-closed-form gap: {worst_ulps} ulp at {worst_where}");
    assert!(
        worst_ulps > 0,
        "the two providers were expected to differ by at least 1 ulp"
    );
}

#[test]
fn table_and_closed_form_have_different_rule_identities() {
    // Intentional inequality: equal identities would let recoupling data
    // computed from the pinned decimals be reused for the closed-form
    // coefficients, which are different numbers.
    assert_ne!(table().rule_identity(), FibonacciFusionRule.rule_identity());
    // ...while two loads of the same pinned bytes are one identity.
    assert_eq!(table().rule_identity(), table().rule_identity());
}

// -------------------------------------------------------------------------
// 3. Consistency checks — never generators
// -------------------------------------------------------------------------

/// Admissible `e` for `F(a,b,c,d,·,·)`: `e ∈ a⊗b` with `e⊗c ∋ d`.
fn left_channels(
    fib: &CategoryDataFibonacci,
    a: SectorId,
    b: SectorId,
    c: SectorId,
    d: SectorId,
) -> Vec<SectorId> {
    fib.fusion_channels(a, b)
        .into_iter()
        .filter(|&e| fib.nsymbol(e, c, d) > 0)
        .collect()
}

/// Admissible `f` for `F(a,b,c,d,·,·)`: `f ∈ b⊗c` with `a⊗f ∋ d`.
fn right_channels(
    fib: &CategoryDataFibonacci,
    a: SectorId,
    b: SectorId,
    c: SectorId,
    d: SectorId,
) -> Vec<SectorId> {
    fib.fusion_channels(b, c)
        .into_iter()
        .filter(|&f| fib.nsymbol(a, f, d) > 0)
        .collect()
}

#[test]
fn f_blocks_are_unitary() {
    let fib = table();
    for a in SECTORS {
        for b in SECTORS {
            for c in SECTORS {
                for d in SECTORS {
                    let rows = left_channels(&fib, a, b, c, d);
                    let columns = right_channels(&fib, a, b, c, d);
                    assert_eq!(rows.len(), columns.len(), "F block is not square");
                    for (i, &f1) in columns.iter().enumerate() {
                        for (j, &f2) in columns.iter().enumerate() {
                            let entry: Complex64 = rows
                                .iter()
                                .map(|&e| {
                                    fib.f_symbol_scalar(a, b, c, d, e, f1).conj()
                                        * fib.f_symbol_scalar(a, b, c, d, e, f2)
                                })
                                .sum();
                            let expected = Complex64::new(f64::from(u8::from(i == j)), 0.0);
                            assert!(
                                (entry - expected).norm() <= FP_BOUND,
                                "F block {:?} not unitary at [{i},{j}]: {entry}",
                                (a, b, c, d)
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn r_symbols_are_unit_modulus_on_admissible_channels() {
    let fib = table();
    for a in SECTORS {
        for b in SECTORS {
            for c in fib.fusion_channels(a, b) {
                let modulus = fib.r_symbol_scalar(a, b, c).norm();
                assert!(
                    (modulus - 1.0).abs() <= FP_BOUND,
                    "|R{:?}| = {modulus}",
                    (a, b, c)
                );
            }
        }
    }
}

#[test]
fn unit_associators_are_bitwise_one() {
    // The three unitor associators of TensorKitSectors' `triangle_equation`
    // (sectors.jl), but asserted bitwise rather than within `FP_BOUND`.
    //
    // This is the one consistency check in this file that is really a
    // *contract* check: `CanonicalUnitFusionRule`
    // (tenet-sectors/src/algebra.rs) promises that unitors and associators act
    // as the identity on every multiplicity space, and the provider implements
    // that marker unconditionally. "Identity" there is exact, not
    // approximate — the engine's Unique/SymmetricBraiding lowering may drop
    // these factors entirely rather than multiply by them. A future table
    // revision whose unit-associated F drifted by one ulp would still satisfy
    // a tolerance, while making the marker a false statement and the lowering
    // silently wrong. So this is the assertion that has to be bitwise.
    let fib = table();
    let unit = fib.vacuum();
    let one = Complex64::new(1.0, 0.0);
    for a in SECTORS {
        for b in SECTORS {
            for c in fib.fusion_channels(a, b) {
                for (which, value) in [
                    ("left unitor", fib.f_symbol_scalar(unit, a, b, c, a, c)),
                    ("middle unitor", fib.f_symbol_scalar(a, unit, b, c, a, b)),
                    ("right unitor", fib.f_symbol_scalar(a, b, unit, c, c, b)),
                ] {
                    assert_bits(value, one, &format!("{which}{:?}", (a, b, c)));
                }
            }
        }
    }
}

#[test]
fn pentagon_equation_holds() {
    // Verbatim structure of TensorKitSectors `pentagon_equation`
    // (sectors.jl:787-820), multiplicity-free branch. Read as a residual
    // check: neither side is ever solved for an unknown F.
    let fib = table();
    let mut cases = 0usize;
    for a in SECTORS {
        for b in SECTORS {
            for c in SECTORS {
                for d in SECTORS {
                    for f in fib.fusion_channels(a, b) {
                        for h in fib.fusion_channels(c, d) {
                            for g in fib.fusion_channels(f, c) {
                                for i in fib.fusion_channels(b, h) {
                                    for e in SECTORS {
                                        if fib.nsymbol(g, d, e) == 0 || fib.nsymbol(a, i, e) == 0 {
                                            continue;
                                        }
                                        let lhs = fib.f_symbol_scalar(f, c, d, e, g, h)
                                            * fib.f_symbol_scalar(a, b, h, e, f, i);
                                        let rhs: Complex64 = fib
                                            .fusion_channels(b, c)
                                            .into_iter()
                                            .map(|j| {
                                                fib.f_symbol_scalar(a, b, c, g, f, j)
                                                    * fib.f_symbol_scalar(a, j, d, e, g, i)
                                                    * fib.f_symbol_scalar(b, c, d, i, j, h)
                                            })
                                            .sum();
                                        assert!(
                                            (lhs - rhs).norm() <= FP_BOUND,
                                            "pentagon{:?}: {lhs} vs {rhs}",
                                            (a, b, c, d, e, f, g, h, i)
                                        );
                                        cases += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(cases > 0, "the pentagon loop never ran");
}

#[test]
fn both_hexagon_equations_hold() {
    // Verbatim structure of TensorKitSectors `hexagon_equation`
    // (sectors.jl:835-869), multiplicity-free branch.
    let fib = table();
    let mut cases = 0usize;
    for a in SECTORS {
        for b in SECTORS {
            for c in SECTORS {
                for e in fib.fusion_channels(c, a) {
                    for f in fib.fusion_channels(c, b) {
                        for d in SECTORS {
                            if fib.nsymbol(e, b, d) == 0 || fib.nsymbol(a, f, d) == 0 {
                                continue;
                            }
                            let facbdef = fib.f_symbol_scalar(a, c, b, d, e, f);
                            let over = fib.r_symbol_scalar(c, a, e)
                                * facbdef
                                * fib.r_symbol_scalar(c, b, f);
                            // `conj(R)` where TensorKitSectors writes
                            // `inv(R)`: equal only because R is unitary, which
                            // `r_symbols_are_unit_modulus_on_admissible_channels`
                            // asserts independently. Not circular — that test
                            // reads R alone and never touches the hexagon.
                            let under = fib.r_symbol_scalar(a, c, e).conj()
                                * facbdef
                                * fib.r_symbol_scalar(b, c, f).conj();
                            let mut path_over = Complex64::new(0.0, 0.0);
                            let mut path_under = Complex64::new(0.0, 0.0);
                            for g in fib.fusion_channels(a, b) {
                                let fcabdeg = fib.f_symbol_scalar(c, a, b, d, e, g);
                                let fabcdgf = fib.f_symbol_scalar(a, b, c, d, g, f);
                                path_over += fcabdeg * fib.r_symbol_scalar(c, g, d) * fabcdgf;
                                path_under +=
                                    fcabdeg * fib.r_symbol_scalar(g, c, d).conj() * fabcdgf;
                            }
                            assert!(
                                (over - path_over).norm() <= FP_BOUND,
                                "hexagon (over){:?}: {over} vs {path_over}",
                                (a, b, c, d, e, f)
                            );
                            assert!(
                                (under - path_under).norm() <= FP_BOUND,
                                "hexagon (under){:?}: {under} vs {path_under}",
                                (a, b, c, d, e, f)
                            );
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(cases > 0, "the hexagon loop never ran");
}

#[test]
fn ribbon_relation_ties_twists_to_the_braid() {
    // theta_c = theta_a * theta_b * R^{ba}_c * R^{ab}_c for every admissible
    // channel: the braid relation the twist derivation must be consistent
    // with. Checked, not used to define the twist.
    let fib = table();
    for a in SECTORS {
        for b in SECTORS {
            for c in fib.fusion_channels(a, b) {
                let monodromy = fib.r_symbol_scalar(b, a, c) * fib.r_symbol_scalar(a, b, c);
                let expected = fib.twist_scalar(c);
                let actual = fib.twist_scalar(a) * fib.twist_scalar(b) * monodromy;
                assert!(
                    (actual - expected).norm() <= FP_BOUND,
                    "ribbon{:?}: {actual} vs {expected}",
                    (a, b, c)
                );
            }
        }
    }
}

// -------------------------------------------------------------------------
// Provider contract
// -------------------------------------------------------------------------

#[test]
fn codec_round_trips_every_object_and_every_admissible_channel() {
    let fib = table();
    for object in fib.objects() {
        let id = fib.encode_sector(&object).expect("object encodes");
        assert_eq!(fib.decode_sector(id).expect("id decodes"), object);
    }
    // Decode totality over the reachable algebra: the vacuum, every dual, and
    // every generated channel must decode.
    fib.decode_sector(fib.vacuum()).expect("vacuum decodes");
    for a in SECTORS {
        fib.decode_sector(fib.dual(a)).expect("dual decodes");
        for b in SECTORS {
            for channel in fib.fusion_channels(a, b) {
                let object = fib.decode_sector(channel).expect("channel decodes");
                assert_eq!(fib.encode_sector(&object).expect("re-encodes"), channel);
            }
        }
    }
    // Labels outside the frozen ordering are rejected, not wrapped.
    for label in [0u16, 3, u16::MAX] {
        assert!(matches!(
            fib.encode_sector(&CategoryObject(label)),
            Err(FusionAlgebraError::UnrepresentableSectorLabel { .. })
        ));
    }
    assert!(matches!(
        fib.decode_sector(SectorId::new(2)),
        Err(FusionAlgebraError::InvalidSector { .. })
    ));
}

#[test]
fn checked_and_infallible_fusion_queries_agree() {
    let fib = table();
    for a in SECTORS {
        assert_eq!(fib.try_dual_sector(a), Ok(fib.dual(a)));
        for b in SECTORS {
            assert_eq!(
                fib.try_fusion_channels(a, b).unwrap().as_slice(),
                fib.fusion_channels(a, b).as_slice()
            );
            for c in SECTORS {
                assert_eq!(fib.try_nsymbol(a, b, c), Ok(fib.nsymbol(a, b, c)));
            }
        }
    }
    // Out-of-domain ids are rejected on the checked path.
    let invalid = SectorId::new(2);
    assert_eq!(
        fib.try_dual_sector(invalid),
        Err(FusionAlgebraError::InvalidSector { sector: invalid })
    );
    assert_eq!(
        fib.try_fusion_channels(invalid, SECTORS[0]),
        Err(FusionAlgebraError::InvalidSector { sector: invalid })
    );
    assert_eq!(
        fib.try_nsymbol(SECTORS[0], SECTORS[0], invalid),
        Err(FusionAlgebraError::InvalidSector { sector: invalid })
    );
}
