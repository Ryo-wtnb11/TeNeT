//! Advanced linear-algebra oracle for the single-precision payloads (#1459).
//!
//! Every operation `AdvancedLinalgScalar` admits — `inv`, `solve`,
//! `solve_right`, `pinv`, `exp` (Hermitian spectral route and non-Hermitian
//! Padé route), `powi`, `sqrt`, `eig_vals`, `eig_full`, `eig_trunc` — is
//! compared against the `f64`/`Complex64` result of the same operation on the
//! exactly widened input (`single_precision_oracle`).
//!
//! Tolerance: `K * sqrt(n) * eps(f32) * max(1, scale) * kappa`, `K = 32`,
//! `n` the payload length. `kappa` multiplies the bound only for a forward
//! error and is **measured** on the double-precision twin:
//!
//! | quantity | `kappa` |
//! | --- | --- |
//! | `inv`, `solve`, `solve_right`, `pinv`, `powi(-p)` | `sigma_max / sigma_min` of the divisor |
//! | `exp`, `powi(+p)`, `sqrt`, `eig_full` residual | 1 (backward stable / elementwise) |
//! | eigenvalues | `sigma_max / sigma_min` of the double-precision eigenvector factor (Bauer–Fike) |

mod single_precision_oracle;

use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use tenet::core::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use tenet::prelude::{GradedSpace, SectorSpectrum, TensorMap, Truncation};

use single_precision_oracle::{
    assert_payloads_agree_scaled, assert_scalars_agree_scaled, fermion_su2_leg_with, minus_one,
    one, runtime, tolerance, u1_leg_with, Parts,
};

/// Hermitian, well separated: diagonal `8, 4, 2, 1` with a small constant
/// off-diagonal. `exp` takes the spectral route on it.
fn hermitian_entry(row: usize, col: usize) -> (f32, f32) {
    const DIAGONAL: [f32; 4] = [8.0, 4.0, 2.0, 1.0];
    if row == col {
        (DIAGONAL[row % 4], 0.0)
    } else if row < col {
        (0.25, 0.125)
    } else {
        (0.25, -0.125)
    }
}

/// Non-Hermitian with distinct eigenvalues: `exp` takes the Padé route and
/// `eig_full` has a well-conditioned eigenbasis.
fn nonnormal_entry(row: usize, col: usize) -> (f32, f32) {
    const DIAGONAL: [f32; 4] = [1.0, 0.5, -0.5, 0.25];
    if row == col {
        (DIAGONAL[row % 4], 0.0)
    } else if row < col {
        (0.25, 0.125)
    } else {
        (-0.125, 0.0625)
    }
}

fn su2_leg_with(degeneracies: [usize; 3]) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), degeneracies[0]),
            (SU2Irrep::from_twice_spin(1), degeneracies[1]),
            (SU2Irrep::from_twice_spin(2), degeneracies[2]),
        ],
    )
    .unwrap()
}

/// `sigma_max / sigma_min` over every coupled block of a double-precision
/// tensor, asserted full rank and below `1e4` so the fixture is a usable
/// single-precision oracle.
macro_rules! measured_kappa {
    ($tensor:expr) => {{
        let mut largest = 0.0f64;
        let mut smallest = f64::INFINITY;
        for entry in &$tensor.svd_vals().unwrap() {
            for &value in &entry.values {
                largest = largest.max(value);
                smallest = smallest.min(value);
            }
        }
        let kappa = largest / smallest;
        assert!(
            smallest > 0.0 && kappa < 1e4,
            "fixture conditioning {kappa:e} is unusable as a single-precision oracle"
        );
        kappa
    }};
}

/// `‖actual - expected‖ <= tolerance(terms, ‖expected‖)`.
macro_rules! assert_residual {
    ($what:expr, $actual:expr, $expected:expr, $terms:expr, $d:ty) => {{
        let expected = $expected;
        let residual = $actual
            .add(expected, one::<$d>(), minus_one::<$d>())
            .unwrap()
            .norm()
            .unwrap();
        let bound = tolerance($terms, expected.norm().unwrap());
        assert!(
            residual <= bound,
            "{}: residual {residual:e} exceeds tolerance {bound:e}",
            $what
        );
    }};
}

/// Complex spectra sector by sector, each value within the bound scaled by
/// `kappa`.
macro_rules! assert_complex_spectra_agree {
    ($what:expr, $narrow:expr, $wide:expr, $terms:expr, $kappa:expr) => {{
        let (narrow, wide) = ($narrow, $wide);
        assert_eq!(narrow.len(), wide.len(), "{}: sector count", $what);
        for (got, expected) in narrow.iter().zip(wide.iter()) {
            assert_eq!(got.sector, expected.sector, "{}: sector order", $what);
            assert_eq!(got.values.len(), expected.values.len(), "{}", $what);
            for (index, (value, oracle)) in got.values.iter().zip(&expected.values).enumerate() {
                assert_scalars_agree_scaled(
                    &format!("{}: sector {:?} value {index}", $what, got.sector),
                    Parts::wide(*value),
                    Parts::wide(*oracle),
                    $terms,
                    $kappa,
                );
            }
        }
    }};
}

/// The whole admitted family over one provider and one payload dtype.
///
/// `$eig` is `D::Eig` and `$to_eig` embeds the narrow input in it; the typed
/// `let` bindings are the static proof that single-precision eig returns
/// `Complex32`, not a widened `Complex64`.
macro_rules! advanced_checks {
    (
        $name:expr, $narrow:ty, $wide:ty, $eig:ty, $to_eig:expr,
        $square:expr, $tall:expr, $min_blocks:expr, $multiplicity_free:expr
    ) => {{
        let rt = runtime();
        let name: &str = $name;
        let square = $square;
        let tall = $tall;
        let (h, wide_h) = twin_with!(
            &rt,
            $narrow,
            $wide,
            [&square],
            [&square],
            |i: &[usize]| { hermitian_entry(i[0], i[1]) }
        );
        let (a, wide_a) = twin_with!(
            &rt,
            $narrow,
            $wide,
            [&square],
            [&square],
            |i: &[usize]| { nonnormal_entry(i[0], i[1]) }
        );
        let (b, wide_b) = twin_with!(
            &rt,
            $narrow,
            $wide,
            [&square],
            [&square],
            |i: &[usize]| { nonnormal_entry(i[1], i[0]) }
        );
        let (t, wide_t) = twin_with!(&rt, $narrow, $wide, [&tall], [&square], |i: &[usize]| {
            hermitian_entry(i[0], i[1])
        });
        let n = wide_h.data().len();
        assert!(
            h.block_count() >= $min_blocks,
            "{name}: the fixture must carry at least {} coupled blocks",
            $min_blocks
        );
        let kappa_h = measured_kappa!(&wide_h);
        let kappa_a = measured_kappa!(&wide_a);
        let kappa_t = measured_kappa!(&wide_t);

        // ---- inv / solve / solve_right / pinv: forward errors. -------------
        assert_payloads_agree_scaled(
            &format!("{name}: inv (kappa {kappa_a:e})"),
            a.inv().unwrap().data(),
            wide_a.inv().unwrap().data(),
            n,
            kappa_a,
        );
        assert_payloads_agree_scaled(
            &format!("{name}: solve (kappa {kappa_a:e})"),
            a.solve(&b).unwrap().data(),
            wide_a.solve(&wide_b).unwrap().data(),
            n,
            kappa_a,
        );
        assert_payloads_agree_scaled(
            &format!("{name}: solve_right (kappa {kappa_a:e})"),
            b.solve_right(&a).unwrap().data(),
            wide_b.solve_right(&wide_a).unwrap().data(),
            n,
            kappa_a,
        );
        assert_payloads_agree_scaled(
            &format!("{name}: pinv (kappa {kappa_t:e})"),
            t.pinv(1e-4).unwrap().data(),
            wide_t.pinv(1e-4).unwrap().data(),
            wide_t.data().len(),
            kappa_t,
        );

        // ---- exp: spectral (Hermitian) and Padé (non-Hermitian) routes. ----
        assert_payloads_agree_scaled(
            &format!("{name}: exp, Hermitian route"),
            h.exp().unwrap().data(),
            wide_h.exp().unwrap().data(),
            n,
            1.0,
        );
        assert_payloads_agree_scaled(
            &format!("{name}: exp, Padé route"),
            a.exp().unwrap().data(),
            wide_a.exp().unwrap().data(),
            n,
            1.0,
        );

        // ---- powi: repeated composition; a negative power inverts once. ----
        assert_payloads_agree_scaled(
            &format!("{name}: powi(3)"),
            a.powi(3).unwrap().data(),
            wide_a.powi(3).unwrap().data(),
            3 * n,
            1.0,
        );
        assert_payloads_agree_scaled(
            &format!("{name}: powi(-2) (kappa {kappa_h:e})"),
            h.powi(-2).unwrap().data(),
            wide_h.powi(-2).unwrap().data(),
            2 * n,
            kappa_h * kappa_h,
        );

        // ---- sqrt of a diagonal bond tensor (dense and compact arms). ------
        let (_, s, _) = h.svd_compact().unwrap();
        let (_, wide_s, _) = wide_h.svd_compact().unwrap();
        let root = s.sqrt().unwrap();
        assert_payloads_agree_scaled(
            &format!("{name}: sqrt of a compact spectrum"),
            root.data(),
            wide_s.sqrt().unwrap().data(),
            n,
            1.0,
        );
        assert_residual!(
            format!("{name}: sqrt ∘ sqrt"),
            root.compose(&root).unwrap(),
            &s,
            n,
            $narrow
        );
        let (dense_diagonal, wide_dense_diagonal) = twin_with!(
            &rt,
            $narrow,
            $wide,
            [&square],
            [&square],
            |i: &[usize]| {
                if i[0] == i[1] {
                    hermitian_entry(i[0], i[1])
                } else {
                    (0.0, 0.0)
                }
            }
        );
        assert_payloads_agree_scaled(
            &format!("{name}: sqrt of a dense diagonal"),
            dense_diagonal.sqrt().unwrap().data(),
            wide_dense_diagonal.sqrt().unwrap().data(),
            n,
            1.0,
        );

        // ---- General eig: factors in `D::Eig`, spectra `Complex64`. --------
        let (d, v): (TensorMap<_, $eig>, TensorMap<_, $eig>) = a.eig_full().unwrap();
        let (_, wide_v) = wide_a.eig_full().unwrap();
        let kappa_v = measured_kappa!(&wide_v);
        let to_eig = $to_eig;
        let a_eig: TensorMap<_, $eig> = to_eig(&a);
        let vd = v.compose(&d).unwrap();
        assert_residual!(
            format!("{name}: eig_full a∘v == v∘d"),
            a_eig.compose(&v).unwrap(),
            &vd,
            n,
            $eig
        );
        let wide_values = wide_a.eig_vals().unwrap();
        let values: Vec<SectorSpectrum<_, Complex64>> = a.eig_vals().unwrap();
        assert_complex_spectra_agree!(
            format!("{name}: eig_vals (kappa(V) {kappa_v:e})"),
            &values,
            &wide_values,
            n,
            kappa_v
        );
        assert_complex_spectra_agree!(
            format!("{name}: eig_full d (kappa(V) {kappa_v:e})"),
            &{
                let mut spectra = d.diagview().unwrap();
                spectra.sort_by(|left, right| left.sector.cmp(&right.sector));
                spectra
            },
            &wide_values,
            n,
            kappa_v
        );
        let truncated = a.eig_trunc(&Truncation::rank(2)).unwrap();
        let wide_truncated = wide_a.eig_trunc(&Truncation::rank(2)).unwrap();
        let truncated_d: &TensorMap<_, $eig> = &truncated.d;
        assert_eq!(
            truncated_d.data().len(),
            wide_truncated.d.data().len(),
            "{name}: eig_trunc kept a different number of states than the widened oracle"
        );
        assert!(truncated.error > 0.0, "{name}: eig_trunc discarded nothing");
        assert_scalars_agree_scaled(
            &format!("{name}: eig_trunc error"),
            Complex64::new(truncated.error, 0.0),
            Complex64::new(wide_truncated.error, 0.0),
            n,
            kappa_v,
        );

        if $multiplicity_free {
            // Lazy-adjoint receivers are materialized for the call.
            let lazy = a.adjoint().unwrap();
            assert_payloads_agree_scaled(
                &format!("{name}: inv of a lazy adjoint"),
                lazy.inv().unwrap().data(),
                wide_a.adjoint().unwrap().inv().unwrap().data(),
                n,
                kappa_a,
            );
            assert_payloads_agree_scaled(
                &format!("{name}: exp of a lazy adjoint"),
                lazy.exp().unwrap().data(),
                wide_a.adjoint().unwrap().exp().unwrap().data(),
                n,
                1.0,
            );
        }
    }};
}

macro_rules! advanced_suite {
    ($suite:ident, $narrow:ty, $wide:ty, $eig:ty, $to_eig:expr) => {
        mod $suite {
            use super::*;

            #[test]
            fn u1_advanced_linalg_matches_the_widened_oracle() {
                advanced_checks!(
                    concat!("U(1) ", stringify!($narrow)),
                    $narrow,
                    $wide,
                    $eig,
                    $to_eig,
                    u1_leg_with([2, 3, 2]),
                    u1_leg_with([3, 4, 3]),
                    3,
                    true
                );
            }

            #[test]
            fn su2_advanced_linalg_matches_the_widened_oracle() {
                advanced_checks!(
                    concat!("SU(2) ", stringify!($narrow)),
                    $narrow,
                    $wide,
                    $eig,
                    $to_eig,
                    su2_leg_with([2, 3, 2]),
                    su2_leg_with([3, 4, 3]),
                    3,
                    true
                );
            }

            #[test]
            fn fermion_su2_advanced_linalg_matches_the_widened_oracle() {
                advanced_checks!(
                    concat!("fZ2 x U(1) x SU(2) ", stringify!($narrow)),
                    $narrow,
                    $wide,
                    $eig,
                    $to_eig,
                    fermion_su2_leg_with([2, 2, 1]),
                    fermion_su2_leg_with([3, 3, 2]),
                    3,
                    true
                );
            }
        }
    };
}

advanced_suite!(f32_payload, f32, f64, Complex32, |t: &TensorMap<_, f32>| t
    .to_c32());
advanced_suite!(
    complex32_payload,
    Complex32,
    Complex64,
    Complex32,
    |t: &TensorMap<_, Complex32>| t.clone()
);

/// The reciprocal of a compact `Complex32` entry whose squared modulus
/// underflows or overflows `f32` while the reciprocal itself is
/// representable: `2^-80`, `2^-80 i` and `2^100` invert exactly.
#[test]
fn compact_complex32_reciprocal_does_not_underflow() {
    let rt = runtime();
    let leg =
        GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), 3)]).unwrap();
    let tiny = 2.0f32.powi(-80);
    let diagonal: TensorMap<_, Complex32> = TensorMap::diagonal(
        &rt,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![
                Complex32::new(tiny, 0.0),
                Complex32::new(0.0, tiny),
                Complex32::new(2.0f32.powi(100), 0.0),
            ],
        }],
    )
    .unwrap();
    let expected = [
        Complex32::new(2.0f32.powi(80), 0.0),
        Complex32::new(0.0, -(2.0f32.powi(80))),
        Complex32::new(2.0f32.powi(-100), 0.0),
    ];
    let inverse = diagonal.inv().unwrap();
    assert_eq!(inverse.diagview().unwrap()[0].values, expected, "inv");
    let pseudo = diagonal.pinv(0.0).unwrap();
    assert_eq!(pseudo.diagview().unwrap()[0].values, expected, "pinv");
}

/// Checked-Generic provider: SU(3), outer multiplicity, the checked dispatch.
#[cfg(feature = "racah-generated")]
mod checked_generic {
    use super::*;
    use tenet::typed::SUNFusionRule;

    fn su3_leg(degeneracy: usize) -> GradedSpace<SUNFusionRule> {
        GradedSpace::try_new_with_arc(
            Arc::new(SUNFusionRule::new(3).unwrap()),
            [(vec![2i64, 2], degeneracy)],
        )
        .unwrap()
    }

    macro_rules! checked_generic_suite {
        ($suite:ident, $narrow:ty, $wide:ty, $eig:ty, $to_eig:expr) => {
            mod $suite {
                use super::*;

                #[test]
                fn su3_advanced_linalg_matches_the_widened_oracle() {
                    advanced_checks!(
                        concat!("SU(3) ", stringify!($narrow)),
                        $narrow,
                        $wide,
                        $eig,
                        $to_eig,
                        su3_leg(2),
                        su3_leg(3),
                        1,
                        false
                    );
                }
            }
        };
    }

    checked_generic_suite!(f32_payload, f32, f64, Complex32, |t: &TensorMap<_, f32>| t
        .to_c32());
    checked_generic_suite!(
        complex32_payload,
        Complex32,
        Complex64,
        Complex32,
        |t: &TensorMap<_, Complex32>| t.clone()
    );
}
