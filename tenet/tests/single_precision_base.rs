//! Base-family oracle for the single-precision payloads (#1315).
//!
//! The oracle for every check here is the **`f64`/`Complex64` result of the
//! same operation on the widened input**. Both tensors are filled through
//! `from_block_fn` from one 24-bit pseudo-random stream, so the double tensor
//! holds exactly `f64::from` of every single-precision entry and the widening
//! contributes no error of its own. The double path is a different execution
//! lane with a different dense kernel, and it is the path the existing
//! TensorKit/QSpace conformance suites already validate, so it is independent
//! of the code under test.
//!
//! Tolerance is `K * sqrt(n) * eps(real(D)) * max(1, max|expected|)` with
//! `K = 32` and `n` the number of payload entries the reduction or kernel sums
//! over — the standard bound for a naive length-`n` floating sum, with `K`
//! covering the recoupling coefficients applied on top of it.
//!
//! Both dtypes are driven from one `Runtime`. That is deliberate: the
//! single-precision lanes, their contract workspaces and their coefficient
//! scratch are created on first use next to the double-precision ones, and a
//! `f32` tensor must never read the `f64` lane's converted recoupling matrix
//! for the same structure identity.

use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use tenet::core::{
    product_sector, FermionParityFusionRule, ProductFusionRule, ProductFusionRuleExt,
    SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep, Z2Irrep,
};
use tenet::prelude::{GradedSpace, LegSelection, Runtime, SectorSpectrum, TensorMap};

/// `K` of the `K * sqrt(n) * eps` bound in the module docs.
const K: f64 = 32.0;

fn runtime() -> Runtime {
    Runtime::builder().dense_threads(1).build().unwrap()
}

/// Payload values that a single-precision tensor and its double-precision twin
/// can both hold exactly.
///
/// `parts` takes `f32` components on purpose: `f64::from(re)` is exact, so the
/// twin is the widening of the single-precision tensor entry by entry, with no
/// rounding to account for before the operation under test runs.
trait Parts: Copy {
    fn parts(re: f32, im: f32) -> Self;
    fn wide(self) -> Complex64;
}

impl Parts for f32 {
    fn parts(re: f32, _im: f32) -> Self {
        re
    }
    fn wide(self) -> Complex64 {
        Complex64::new(f64::from(self), 0.0)
    }
}

impl Parts for f64 {
    fn parts(re: f32, _im: f32) -> Self {
        f64::from(re)
    }
    fn wide(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
}

impl Parts for Complex32 {
    fn parts(re: f32, im: f32) -> Self {
        Self::new(re, im)
    }
    fn wide(self) -> Complex64 {
        Complex64::new(f64::from(self.re), f64::from(self.im))
    }
}

impl Parts for Complex64 {
    fn parts(re: f32, im: f32) -> Self {
        Self::new(f64::from(re), f64::from(im))
    }
    fn wide(self) -> Complex64 {
        self
    }
}

/// One 24-bit draw in `[-1, 1)`: exactly representable in `f32`, so the two
/// twins receive the same number and not two roundings of one.
fn draw(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    ((value >> 40) as f32) / ((1_u32 << 23) as f32) - 1.0
}

fn draw_parts<D: Parts>(state: &mut u64) -> D {
    let re = draw(state);
    let im = draw(state);
    D::parts(re, im)
}

/// Builds the same tensor at two precisions on one runtime.
///
/// A macro rather than a function: the payload dtype is the only thing that
/// varies, and the alternative is repeating the eight dispatch bounds of
/// `from_block_fn` for both instantiations.
macro_rules! twin {
    ($rt:expr, $narrow:ty, $wide:ty, $codomain:expr, $domain:expr, $seed:expr) => {{
        let mut state = $seed;
        let narrow: TensorMap<_, $narrow> =
            TensorMap::from_block_fn($rt, $codomain, $domain, |_, _| draw_parts(&mut state))
                .unwrap();
        let mut state = $seed;
        let wide: TensorMap<_, $wide> =
            TensorMap::from_block_fn($rt, $codomain, $domain, |_, _| draw_parts(&mut state))
                .unwrap();
        (narrow, wide)
    }};
}

/// Asserts the two payloads agree within the module's tolerance, and that the
/// twins really did describe the same tensor (equal length, equal blocks).
fn assert_payloads_agree<D: Parts, W: Parts>(what: &str, narrow: &[D], wide: &[W], terms: usize) {
    assert_eq!(
        narrow.len(),
        wide.len(),
        "{what}: the twins hold different payload lengths"
    );
    let scale = wide
        .iter()
        .map(|&value| value.wide().norm())
        .fold(0.0f64, f64::max)
        .max(1.0);
    let tolerance = K * (terms as f64).sqrt() * f64::from(f32::EPSILON) * scale;
    for (index, (&got, &expected)) in narrow.iter().zip(wide).enumerate() {
        let error = (got.wide() - expected.wide()).norm();
        assert!(
            error <= tolerance,
            "{what}: entry {index} is {got:?} against the widened oracle {expected:?} \
             (error {error:e} > tolerance {tolerance:e})",
            got = got.wide(),
            expected = expected.wide(),
        );
    }
}

fn assert_scalars_agree(what: &str, narrow: Complex64, wide: Complex64, terms: usize) {
    let tolerance = K * (terms as f64).sqrt() * f64::from(f32::EPSILON) * wide.norm().max(1.0);
    let error = (narrow - wide).norm();
    assert!(
        error <= tolerance,
        "{what}: {narrow} against the widened oracle {wide} \
         (error {error:e} > tolerance {tolerance:e})"
    );
}

/// U(1): abelian, several blocks, nontrivial degeneracies.
fn u1_leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap()
}

type FermionU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
type FermionSu2Rule = ProductFusionRule<FermionU1Rule, SU2FusionRule>;

/// fZ2 x U(1) x SU(2): fermionic signs, nontrivial braiding and non-abelian
/// recoupling with `dim(c) != 1` in one provider.
fn fermion_su2_leg() -> GradedSpace<FermionSu2Rule> {
    let provider = Arc::new(
        FermionParityFusionRule
            .product(U1FusionRule)
            .product(SU2FusionRule),
    );
    let label = |parity, charge, twice_spin| {
        product_sector(
            product_sector(parity, U1Irrep::new(charge)),
            SU2Irrep::from_twice_spin(twice_spin),
        )
    };
    GradedSpace::try_new_with_arc(
        provider,
        [
            (label(Z2Irrep::EVEN, 0, 0), 2),
            (label(Z2Irrep::ODD, 1, 1), 2),
            (label(Z2Irrep::EVEN, 2, 2), 1),
        ],
    )
    .unwrap()
}

/// Everything below is generated once per admitted single-precision dtype.
macro_rules! base_suite {
    ($suite:ident, $narrow:ty, $wide:ty) => {
        mod $suite {
            use super::*;

            #[test]
            fn arithmetic_and_reductions_match_the_widened_oracle() {
                let runtime = runtime();
                let leg = u1_leg();
                let (a, wa) = twin!(&runtime, $narrow, $wide, [&leg, &leg], [&leg], 9_001);
                let (b, wb) = twin!(&runtime, $narrow, $wide, [&leg, &leg], [&leg], 9_002);
                let terms = wa.data().len();

                assert_payloads_agree("input", a.data(), wa.data(), 1);

                let scale = <$narrow as Parts>::parts(0.5, -0.25);
                let wide_scale = <$wide as Parts>::parts(0.5, -0.25);
                assert_payloads_agree(
                    "scale",
                    a.scale(scale).data(),
                    wa.scale(wide_scale).data(),
                    1,
                );
                assert_payloads_agree(
                    "add",
                    a.add(&b, scale, wide_of_one::<$narrow>()).unwrap().data(),
                    wa.add(&wb, wide_scale, wide_of_one::<$wide>())
                        .unwrap()
                        .data(),
                    2,
                );
                assert_payloads_agree(
                    "normalize",
                    a.normalize().unwrap().data(),
                    wa.normalize().unwrap().data(),
                    terms,
                );
                assert_payloads_agree(
                    "zeros_like",
                    a.zeros_like().data(),
                    wa.zeros_like().data(),
                    1,
                );

                assert_scalars_agree(
                    "norm",
                    Complex64::new(a.norm().unwrap(), 0.0),
                    Complex64::new(wa.norm().unwrap(), 0.0),
                    terms,
                );
                assert_scalars_agree(
                    "norm_inf",
                    Complex64::new(a.norm_inf().unwrap(), 0.0),
                    Complex64::new(wa.norm_inf().unwrap(), 0.0),
                    1,
                );
                assert_scalars_agree(
                    "norm_p(3)",
                    Complex64::new(a.norm_p(3.0).unwrap(), 0.0),
                    Complex64::new(wa.norm_p(3.0).unwrap(), 0.0),
                    terms,
                );
                assert_scalars_agree(
                    "inner",
                    a.inner(&b).unwrap().wide(),
                    wa.inner(&wb).unwrap().wide(),
                    terms,
                );

                let square = a.compose(&a.adjoint().unwrap()).unwrap();
                let wide_square = wa.compose(&wa.adjoint().unwrap()).unwrap();
                assert_scalars_agree(
                    "tr",
                    square.tr().unwrap().wide(),
                    wide_square.tr().unwrap().wide(),
                    terms,
                );
            }

            #[test]
            fn structural_transforms_match_the_widened_oracle() {
                let runtime = runtime();
                let leg = fermion_su2_leg();
                let (a, wa) = twin!(&runtime, $narrow, $wide, [&leg, &leg], [&leg, &leg], 9_101);
                let terms = wa.data().len();

                for (what, got, expected) in [
                    (
                        "permute",
                        a.permute(&[1, 2], &[3, 0]).unwrap(),
                        wa.permute(&[1, 2], &[3, 0]).unwrap(),
                    ),
                    (
                        "braid",
                        a.braid(&[1, 2], &[3, 0], &[0, 1, 2, 3]).unwrap(),
                        wa.braid(&[1, 2], &[3, 0], &[0, 1, 2, 3]).unwrap(),
                    ),
                    (
                        "transpose",
                        a.transpose().unwrap(),
                        wa.transpose().unwrap(),
                    ),
                    (
                        "repartition",
                        a.repartition(1).unwrap(),
                        wa.repartition(1).unwrap(),
                    ),
                    ("twist", a.twist(&[0, 2]).unwrap(), wa.twist(&[0, 2]).unwrap()),
                    (
                        "twist_inverse",
                        a.twist_inverse(&[1]).unwrap(),
                        wa.twist_inverse(&[1]).unwrap(),
                    ),
                    ("flip", a.flip(&[0]).unwrap(), wa.flip(&[0]).unwrap()),
                    (
                        "adjoint (lazy)",
                        a.adjoint().unwrap(),
                        wa.adjoint().unwrap(),
                    ),
                    (
                        "adjoint materialized",
                        a.adjoint().unwrap().permute(&[0, 1], &[2, 3]).unwrap(),
                        wa.adjoint().unwrap().permute(&[0, 1], &[2, 3]).unwrap(),
                    ),
                    (
                        "trace_pairs",
                        a.trace_pairs(&[(0, 3)]).unwrap(),
                        wa.trace_pairs(&[(0, 3)]).unwrap(),
                    ),
                ] {
                    assert_payloads_agree(what, got.data(), expected.data(), terms);
                    assert_eq!(
                        got.block_count(),
                        expected.block_count(),
                        "{what}: block structure diverged between the twins"
                    );
                }

                // Negative control: the fixture is not invariant under the
                // transforms above, so agreement is evidence and not an
                // accident of a payload the operations leave alone.
                let moved = a.permute(&[1, 2], &[3, 0]).unwrap();
                let largest = wa
                    .data()
                    .iter()
                    .zip(moved.data())
                    .map(|(&expected, &got)| (got.wide() - expected.wide()).norm())
                    .fold(0.0f64, f64::max);
                assert!(
                    largest > 1e-3,
                    "the SU(2)/fermionic fixture must be moved by permute, largest change {largest:e}"
                );
                assert_ne!(
                    a.twist(&[0, 2]).unwrap().data(),
                    a.data(),
                    "the fZ2 twist must change the payload of an odd-sector fixture"
                );

                // Round trip: the inverse permutation restores the input
                // exactly, because the recoupling of a permutation and its
                // inverse is the identity matrix at any payload precision.
                let restored = a.permute(&[1, 2], &[3, 0]).unwrap();
                assert_payloads_agree(
                    "permute round trip",
                    restored.permute(&[3, 0], &[1, 2]).unwrap().data(),
                    a.data(),
                    terms,
                );
            }

            #[test]
            fn products_and_contractions_match_the_widened_oracle() {
                let runtime = runtime();
                let leg = fermion_su2_leg();
                let (a, wa) = twin!(&runtime, $narrow, $wide, [&leg], [&leg], 9_201);
                let (b, wb) = twin!(&runtime, $narrow, $wide, [&leg], [&leg], 9_202);
                let terms = wa.data().len();

                for (what, got, expected) in [
                    (
                        "contract",
                        a.contract(&b, &[1], &[0], &[0, 1]).unwrap(),
                        wa.contract(&wb, &[1], &[0], &[0, 1]).unwrap(),
                    ),
                    ("compose", a.compose(&b).unwrap(), wa.compose(&wb).unwrap()),
                    ("otimes", a.otimes(&b).unwrap(), wa.otimes(&wb).unwrap()),
                    (
                        "catdomain",
                        a.catdomain(&b).unwrap(),
                        wa.catdomain(&wb).unwrap(),
                    ),
                    (
                        "catcodomain",
                        a.catcodomain(&b).unwrap(),
                        wa.catcodomain(&wb).unwrap(),
                    ),
                ] {
                    assert_payloads_agree(what, got.data(), expected.data(), terms);
                }
            }

            #[test]
            fn rank_five_multi_block_contraction_matches_the_widened_oracle() {
                let runtime = runtime();
                let leg = u1_leg();
                let (a, wa) = twin!(
                    &runtime,
                    $narrow,
                    $wide,
                    [&leg, &leg, &leg],
                    [&leg, &leg],
                    9_301
                );
                assert_eq!(a.rank(), 5);
                assert!(a.block_count() > 1, "the fixture must be multi-block");
                let (b, wb) = twin!(&runtime, $narrow, $wide, [&leg, &leg], [&leg], 9_302);
                let terms = wa.data().len().max(wb.data().len());

                assert_payloads_agree(
                    "rank-5 contract",
                    a.contract(&b, &[3, 4], &[0, 1], &[0, 1, 2, 3])
                        .unwrap()
                        .data(),
                    wa.contract(&wb, &[3, 4], &[0, 1], &[0, 1, 2, 3])
                        .unwrap()
                        .data(),
                    terms,
                );
                assert_payloads_agree(
                    "rank-5 permute",
                    a.permute(&[4, 0, 2], &[1, 3]).unwrap().data(),
                    wa.permute(&[4, 0, 2], &[1, 3]).unwrap().data(),
                    terms,
                );
            }

            #[test]
            fn leg_restriction_and_diagonal_views_match_the_widened_oracle() {
                let runtime = runtime();
                let leg = u1_leg();
                let (a, wa) = twin!(&runtime, $narrow, $wide, [&leg], [&leg], 9_401);
                let terms = wa.data().len();

                let selection =
                    LegSelection::try_new(&leg, [(U1Irrep::new(0), 0..2), (U1Irrep::new(1), 0..1)])
                        .unwrap();
                let restricted = a.restrict_leg(1, &selection).unwrap();
                let wide_restricted = wa.restrict_leg(1, &selection).unwrap();
                assert_payloads_agree(
                    "restrict_leg",
                    restricted.data(),
                    wide_restricted.data(),
                    1,
                );
                assert_payloads_agree(
                    "embed_leg",
                    restricted.embed_leg(1, &selection).unwrap().data(),
                    wide_restricted.embed_leg(1, &selection).unwrap().data(),
                    1,
                );

                // Compact diagonal storage: an operand that never materializes
                // its off-diagonal zeros.
                // Compact diagonal storage: an operand that never
                // materializes its off-diagonal zeros.
                let compact: TensorMap<U1FusionRule, $narrow> =
                    TensorMap::diagonal(&runtime, &leg, diagonal_spectra(9_402)).unwrap();
                let wide_compact: TensorMap<U1FusionRule, $wide> =
                    TensorMap::diagonal(&runtime, &leg, diagonal_spectra(9_402)).unwrap();

                for (got, expected) in compact
                    .diagview()
                    .unwrap()
                    .iter()
                    .zip(wide_compact.diagview().unwrap().iter())
                {
                    assert_eq!(got.sector, expected.sector);
                    assert_payloads_agree("diagview", &got.values, &expected.values, 1);
                }
                assert_scalars_agree(
                    "compact norm",
                    Complex64::new(compact.norm().unwrap(), 0.0),
                    Complex64::new(wide_compact.norm().unwrap(), 0.0),
                    terms,
                );
                assert_scalars_agree(
                    "compact tr",
                    compact.tr().unwrap().wide(),
                    wide_compact.tr().unwrap().wide(),
                    terms,
                );
                assert_payloads_agree(
                    "restrict_diagonal",
                    compact.restrict_diagonal(&selection).unwrap().data(),
                    wide_compact.restrict_diagonal(&selection).unwrap().data(),
                    1,
                );
                assert_payloads_agree(
                    "compact compose",
                    compact.compose(&a).unwrap().data(),
                    wide_compact.compose(&wa).unwrap().data(),
                    terms,
                );
            }

            /// The recoupling matrix of one structure identity is converted
            /// into the lane's own workspace scratch. Running both precisions
            /// against the same structure on one runtime, interleaved and
            /// repeated, fails if either lane reads the other's converted
            /// scratch.
            #[test]
            fn single_and_double_lanes_do_not_share_coefficient_scratch() {
                let runtime = runtime();
                let leg = fermion_su2_leg();
                let (a, wa) = twin!(&runtime, $narrow, $wide, [&leg, &leg], [&leg, &leg], 9_501);
                let terms = wa.data().len();
                for _ in 0..3 {
                    let narrow = a.permute(&[2, 0], &[1, 3]).unwrap();
                    let wide = wa.permute(&[2, 0], &[1, 3]).unwrap();
                    assert_payloads_agree(
                        "interleaved permute",
                        narrow.data(),
                        wide.data(),
                        terms,
                    );
                }
            }
        }
    };
}

/// The compact diagonal fixture on [`u1_leg`], at either precision.
fn diagonal_spectra<D: Parts>(mut state: u64) -> [SectorSpectrum<U1Irrep, D>; 3] {
    [
        (U1Irrep::new(-1), 2),
        (U1Irrep::new(0), 3),
        (U1Irrep::new(1), 2),
    ]
    .map(|(sector, degeneracy)| SectorSpectrum {
        sector,
        values: (0..degeneracy).map(|_| draw_parts(&mut state)).collect(),
    })
}

/// Multiplicative unit of a payload dtype, for the `beta` of `add`.
fn wide_of_one<D: Parts>() -> D {
    D::parts(1.0, 0.0)
}

base_suite!(f32_payload, f32, f64);
base_suite!(complex32_payload, Complex32, Complex64);

/// Checked-Generic providers: outer multiplicity, real structural
/// coefficients, and a recoupling matrix per vertex pattern.
///
/// SU(3) with the adjoint irrep produces fusion trees that differ only by
/// their multiplicity vertex, which the multiplicity-free fixtures above
/// cannot reach.
#[cfg(feature = "racah-generated")]
mod checked_generic {
    use super::*;
    use tenet::typed::SUNFusionRule;

    macro_rules! checked_generic_suite {
        ($suite:ident, $narrow:ty, $wide:ty) => {
            mod $suite {
                use super::*;

                #[test]
                fn su3_base_operations_match_the_widened_oracle() {
                    let runtime = runtime();
                    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
                    let adjoint = vec![2i64, 2];
                    let leg = GradedSpace::try_new_with_arc(
                        Arc::clone(&provider),
                        [(adjoint.clone(), 2)],
                    )
                    .unwrap();
                    let (a, wa) =
                        twin!(&runtime, $narrow, $wide, [&leg, &leg], [&leg, &leg], 9_601);
                    assert!(
                        (0..a.block_count()).any(|index| a
                            .block_fusion_trees(index)
                            .unwrap()
                            .codomain_vertices()
                            .iter()
                            .any(|vertex| vertex.get() > 1)),
                        "the fixture must carry a Generic multiplicity vertex"
                    );
                    let terms = wa.data().len();

                    for (what, got, expected) in [
                        (
                            "permute",
                            a.permute(&[2, 0], &[1, 3]).unwrap(),
                            wa.permute(&[2, 0], &[1, 3]).unwrap(),
                        ),
                        (
                            "repartition",
                            a.repartition(3).unwrap(),
                            wa.repartition(3).unwrap(),
                        ),
                        ("adjoint", a.adjoint().unwrap(), wa.adjoint().unwrap()),
                        (
                            "trace_pairs",
                            a.trace_pairs(&[(0, 3)]).unwrap(),
                            wa.trace_pairs(&[(0, 3)]).unwrap(),
                        ),
                    ] {
                        assert_payloads_agree(what, got.data(), expected.data(), terms);
                    }

                    assert_scalars_agree(
                        "norm",
                        Complex64::new(a.norm().unwrap(), 0.0),
                        Complex64::new(wa.norm().unwrap(), 0.0),
                        terms,
                    );
                    assert_scalars_agree(
                        "inner",
                        a.inner(&a).unwrap().wide(),
                        wa.inner(&wa).unwrap().wide(),
                        terms,
                    );
                }
            }
        };
    }

    checked_generic_suite!(f32_payload, f32, f64);
    checked_generic_suite!(complex32_payload, Complex32, Complex64);
}
