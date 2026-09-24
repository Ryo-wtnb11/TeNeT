//! Truncation as a composition (#1300).
//!
//! The invariant: `svd_compact` → `diagview` → `find_truncated` →
//! `restrict_leg`/`restrict_diagonal` reproduces `svd_trunc`, and
//! `eigh_full` → … reproduces `eigh_trunc`. The oracle for the decision is
//! Host `svd_trunc`/`eigh_trunc` itself, which is the point: the primitives
//! are only useful if they make the very decision and the very slicing those
//! two make.
//!
//! Each case asserts in this order, because a payload comparison over two
//! different layouts would be meaningless:
//!
//! 1. **spaces and layout** — exact: every factor's codomain/domain, then
//!    block count and per-block key, shape, strides and offset. Host builds
//!    the truncated space with `derive_from_final_homspace`, the composition
//!    with `build_root`; nothing guarantees a priori that the two agree. The
//!    kept bond subspace (which values survive, in which sector) is
//!    combinatorial and also exact.
//! 2. **payloads** — within the workspace tolerance rule
//!    (`docs/testing_numerics.md`). Both routes factorize the same input and
//!    then only copy; that the two routes agree is the contract, so they are
//!    compared with each other, and a factorization kernel that reorders its
//!    arithmetic does not break it.
//! 3. **truncation error** — within the tolerance of both Host's error and an
//!    independent oracle, `sqrt(sum_c dim(c) sum_discarded |v|^2)` summed here
//!    from the offered spectrum and the kept prefix counts.
//!
//! ## Cross-sector exact ties
//!
//! `select_truncation` breaks an exact tie between two sectors in ascending
//! `SectorId` order (a deterministic TeNeT rule, not TensorKit's `isless`
//! order for every sector type), sorting the
//! feed itself when a producer hands it another order (#1305). Host feeds the
//! first-encounter block order of the input and `find_truncated` a sorted
//! one, so the tie cases below are part of the exact selection gate whatever order a
//! `BlockStructure` stores its blocks in.

use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use tenet::core::{
    product_sector, FermionParityFusionRule, ProductFusionRuleExt, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
use tenet::prelude::{Error, Runtime, TensorMap, Truncation};
use tenet::typed::{
    GradedSpace, LegSelection, SectorSpectrum, SpectrumMagnitude, TruncatedSelection,
};

#[path = "../../tests/support/numerics.rs"]
mod numerics;

fn runtime() -> Runtime {
    Runtime::builder().dense_threads(1).build().unwrap()
}

/// Asserts that the fixture really is degenerate: every value of every sector
/// is the *same bit pattern*, so a backend change cannot silently de-tie the
/// spectrum and turn the tie sweeps below into ordinary cuts.
macro_rules! assert_every_value_is_the_same_bit_pattern {
    ($tensor:expr) => {{
        let spectra = $tensor.diagview().unwrap();
        let mut all = spectra.iter().flat_map(|entry| entry.values.iter());
        let first = all.next().expect("a nonempty spectrum").to_bits();
        assert!(
            spectra
                .iter()
                .flat_map(|entry| entry.values.iter())
                .all(|value| value.to_bits() == first),
            "the fixture must be exactly degenerate for a tie test, got {spectra:?}"
        );
        assert!(
            spectra.len() > 1,
            "a cross-sector tie needs several sectors"
        );
    }};
}

/// Deterministic structureless fill, so no policy can accidentally reproduce a
/// pattern in the spectrum.
fn fill(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as f64) / (u32::MAX as f64) - 0.5
}

/// Every factor's spaces, then its complete block geometry.
macro_rules! assert_same_layout {
    ($got:expr, $want:expr, $what:expr) => {{
        let got = &$got;
        let want = &$want;
        assert_eq!(got.codomain(), want.codomain(), "{} codomain", $what);
        assert_eq!(got.domain(), want.domain(), "{} domain", $what);
        assert_eq!(
            got.block_count(),
            want.block_count(),
            "{} block count",
            $what
        );
        for index in 0..want.block_count() {
            let (left, right) = (got.block(index).unwrap(), want.block(index).unwrap());
            assert_eq!(left.key(), right.key(), "{} block {index} key", $what);
            assert_eq!(left.shape(), right.shape(), "{} block {index} shape", $what);
            assert_eq!(
                left.strides(),
                right.strides(),
                "{} block {index} strides",
                $what
            );
            assert_eq!(
                left.offset(),
                right.offset(),
                "{} block {index} offset",
                $what
            );
        }
    }};
}

/// `dim(c)` of one sector, read from a one-sector space of degeneracy 1.
macro_rules! quantum_dim {
    ($leg:expr, $sector:expr) => {
        GradedSpace::try_new($leg.provider().clone(), [($sector.clone(), 1)])
            .unwrap()
            .dim()
            .unwrap()
    };
}

/// Independent oracle for `TruncatedSelection::error`: the quantum-dimension
/// weighted 2-norm of every offered value past the kept prefix of its sector.
macro_rules! discarded_norm {
    ($bond:expr, $spectra:expr, $selection:expr) => {{
        let kept = $selection.subspace();
        let kept_sectors = kept.sectors().unwrap();
        let mut sum = 0.0f64;
        for entry in $spectra.iter() {
            let prefix = kept_sectors
                .iter()
                .position(|sector| *sector == entry.sector)
                .map_or(0, |index| kept.degeneracies()[index]);
            let weight = quantum_dim!($bond, entry.sector);
            for &value in &entry.values[prefix..] {
                let magnitude = SpectrumMagnitude::magnitude(value);
                sum += weight * magnitude * magnitude;
            }
        }
        sum.sqrt()
    }};
}

/// The error agrees with Host and with [`discarded_norm`]; `terms` is the
/// number of offered values.
macro_rules! assert_truncation_error {
    ($found:expr, $host_error:expr, $bond:expr, $spectra:expr, $case:expr) => {{
        let terms: usize = $spectra.iter().map(|e| e.values.len()).sum();
        let oracle = discarded_norm!($bond, $spectra, $found.selection);
        numerics::assert_close(
            &format!("{}: truncation error against Host", $case),
            $found.error,
            $host_error,
            terms,
        );
        numerics::assert_close(
            &format!(
                "{}: truncation error against the discarded-norm oracle",
                $case
            ),
            $found.error,
            oracle,
            terms,
        );
    }};
}

/// Payloads of two routes that factorize the same input and then only copy.
/// Every source entry of a block can reach every factor entry of it, so
/// `terms` is the source payload length.
macro_rules! assert_same_payload {
    ($got:expr, $want:expr, $terms:expr, $what:expr) => {{
        numerics::assert_slices_close(&$what, $got.data(), $want.data(), $terms);
        let got = $got.diagonal_spectrum().unwrap();
        let want = $want.diagonal_spectrum().unwrap();
        assert_eq!(got.is_some(), want.is_some(), "{} compact storage", $what);
        for (got, want) in got.iter().flatten().zip(want.iter().flatten()) {
            assert_eq!(got.sector, want.sector, "{} compact sector", $what);
            numerics::assert_slices_close(
                &format!("{} compact values", $what),
                &got.values,
                &want.values,
                $terms,
            );
        }
    }};
}

/// The whole gate for one SVD case.
macro_rules! assert_svd_composition {
    ($source:expr, $truncation:expr, $case:expr) => {{
        let source = &$source;
        let truncation: &Truncation = &$truncation;
        let case: &str = $case;

        let (u, s, vh) = source.svd_compact().unwrap();
        let bond = s.domain()[0].clone();
        let spectra = s.diagview().unwrap();
        let found = bond.find_truncated(&spectra, truncation).unwrap();
        let selection = &found.selection;

        let host = source.svd_trunc(truncation).unwrap();

        let got_u = u.restrict_leg(u.codomain_rank(), selection).unwrap();
        let got_s = s.restrict_diagonal(selection).unwrap();
        let got_vh = vh.restrict_leg(0, selection).unwrap();

        assert_eq!(
            *selection.subspace(),
            host.s.domain()[0],
            "{case}: kept bond space"
        );
        assert_same_layout!(got_u, host.u, format!("{case}: u"));
        assert_same_layout!(got_s, host.s, format!("{case}: s"));
        assert_same_layout!(got_vh, host.vh, format!("{case}: vh"));

        let terms = source.data().len();
        assert_same_payload!(got_u, host.u, terms, format!("{case}: u payload"));
        assert_same_payload!(got_s, host.s, terms, format!("{case}: s payload"));
        assert_same_payload!(got_vh, host.vh, terms, format!("{case}: vh payload"));
        assert_truncation_error!(found, host.error, bond, spectra, case);
        let kept: usize = host.singular_values.iter().map(|e| e.values.len()).sum();
        let offered: usize = spectra.iter().map(|e| e.values.len()).sum();
        assert_eq!(
            selection.is_full(),
            kept == offered,
            "{case}: is_full agrees with the kept count"
        );
        found
    }};
}

/// The whole gate for one Hermitian eigendecomposition case.
macro_rules! assert_eigh_composition {
    ($source:expr, $truncation:expr, $case:expr) => {{
        let source = &$source;
        let truncation: &Truncation = &$truncation;
        let case: &str = $case;

        let (d, v) = source.eigh_full().unwrap();
        let bond = d.domain()[0].clone();
        let spectra = d.diagview().unwrap();
        let found = bond.find_truncated(&spectra, truncation).unwrap();
        let selection = &found.selection;

        let host = source.eigh_trunc(truncation).unwrap();

        let got_d = d.restrict_diagonal(selection).unwrap();
        let got_v = v.restrict_leg(v.codomain_rank(), selection).unwrap();

        assert_eq!(
            *selection.subspace(),
            host.d.domain()[0],
            "{case}: kept bond space"
        );
        assert_same_layout!(got_d, host.d, format!("{case}: d"));
        assert_same_layout!(got_v, host.v, format!("{case}: v"));
        let terms = source.data().len();
        assert_same_payload!(got_d, host.d, terms, format!("{case}: d payload"));
        assert_same_payload!(got_v, host.v, terms, format!("{case}: v payload"));
        assert_truncation_error!(found, host.error, bond, spectra, case);
        let kept: usize = host.eigenvalues.iter().map(|e| e.values.len()).sum();
        let offered: usize = spectra.iter().map(|e| e.values.len()).sum();
        assert_eq!(
            selection.is_full(),
            kept == offered,
            "{case}: is_full agrees with the kept count"
        );
        found
    }};
}

fn u1_leg(pairs: &[(i32, usize)]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        pairs
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

fn su2_leg(pairs: &[(usize, usize)]) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(SU2FusionRule),
        pairs
            .iter()
            .map(|&(twice_spin, degeneracy)| (SU2Irrep::from_twice_spin(twice_spin), degeneracy)),
    )
    .unwrap()
}

fn fz2_leg(pairs: &[(bool, usize)]) -> GradedSpace<FermionParityFusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(FermionParityFusionRule),
        pairs
            .iter()
            .map(|&(odd, degeneracy)| (if odd { Z2Irrep::ODD } else { Z2Irrep::EVEN }, degeneracy)),
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// SVD, Host -> Host, every provider, both dtypes, every policy.
// ---------------------------------------------------------------------------

macro_rules! svd_policy_sweep {
    ($source:expr, $target:expr, $tag:expr) => {{
        let source = $source;
        for (name, truncation) in [
            ("Full", Truncation::Full),
            ("Rank(3)", Truncation::rank(3)),
            ("Rank(1)", Truncation::rank(1)),
            ("Rank(0)", Truncation::rank(0)),
            ("Rank(huge)", Truncation::rank(4096)),
            ("Tolerance", Truncation::relative_cutoff(0.25).unwrap()),
            (
                "ToleranceInf",
                Truncation::relative_inf_cutoff(0.4).unwrap(),
            ),
            ("DiscardWeight", Truncation::relative_error(0.2).unwrap()),
            ("Space", Truncation::space($target.truncspace())),
            (
                "Space & Rank",
                Truncation::space($target.truncspace()).and(Truncation::rank(2)),
            ),
        ] {
            assert_svd_composition!(source, truncation, &format!("{} {name}", $tag));
        }
    }};
}

macro_rules! eigh_policy_sweep {
    ($source:expr, $target:expr, $tag:expr) => {{
        let source = $source;
        for (name, truncation) in [
            ("Full", Truncation::Full),
            ("Rank(3)", Truncation::rank(3)),
            ("Rank(1)", Truncation::rank(1)),
            ("Rank(0)", Truncation::rank(0)),
            ("Tolerance", Truncation::relative_cutoff(0.25).unwrap()),
            (
                "ToleranceInf",
                Truncation::relative_inf_cutoff(0.4).unwrap(),
            ),
            ("DiscardWeight", Truncation::relative_error(0.2).unwrap()),
            ("Space", Truncation::space($target.truncspace())),
            (
                "Space & Rank",
                Truncation::space($target.truncspace()).and(Truncation::rank(2)),
            ),
        ] {
            assert_eigh_composition!(source, truncation, &format!("{} {name}", $tag));
        }
    }};
}

#[test]
fn u1_svd_composition_matches_host_for_every_policy() {
    let left = u1_leg(&[(-1, 2), (0, 3), (1, 2)]);
    let right = u1_leg(&[(-1, 3), (0, 2), (1, 3)]);
    let mut state = 0x1234_5678u64;
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime(), [&left], [&right], move |_, _| fill(&mut state))
            .unwrap();
    svd_policy_sweep!(source, u1_leg(&[(-1, 1), (0, 2)]), "u1 f64");
}

#[test]
fn u1_complex_svd_composition_matches_host_for_every_policy() {
    let left = u1_leg(&[(-1, 2), (0, 3), (1, 2)]);
    let right = u1_leg(&[(-1, 3), (0, 2), (1, 3)]);
    let mut state = 0x2222_1111u64;
    let source: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime(), [&left], [&right], move |_, _| {
            Complex64::new(fill(&mut state), fill(&mut state))
        })
        .unwrap();
    svd_policy_sweep!(source, u1_leg(&[(-1, 1), (0, 2)]), "u1 c64");
}

/// The same composition at single precision (#1324).
///
/// `svd_trunc` is `svd_compact` + `find_truncated(diagview(s))` +
/// `restrict_*`: `real_spectrum` widens the payload's singular values into the
/// `f64` the decision runs on, `s` stores `from_real` of those same widened
/// values, and `SpectrumMagnitude` for `f32` / `Complex32` is
/// `f64::from(v).abs()` / `hypot(re, 0)`. Both routes therefore hand the
/// selection the same `f64` slice, so the kept subspace agrees exactly. This
/// is the path the single-precision `SpectrumMagnitude` impls exist for.
#[test]
fn u1_single_precision_svd_composition_matches_host_for_every_policy() {
    let left = u1_leg(&[(-1, 2), (0, 3), (1, 2)]);
    let right = u1_leg(&[(-1, 3), (0, 2), (1, 3)]);
    let mut state = 0x3333_4444u64;
    let source: TensorMap<_, f32> =
        TensorMap::from_block_fn(&runtime(), [&left], [&right], move |_, _| {
            fill(&mut state) as f32
        })
        .unwrap();
    svd_policy_sweep!(source, u1_leg(&[(-1, 1), (0, 2)]), "u1 f32");
}

#[test]
fn u1_complex32_svd_composition_matches_host_for_every_policy() {
    let left = u1_leg(&[(-1, 2), (0, 3), (1, 2)]);
    let right = u1_leg(&[(-1, 3), (0, 2), (1, 3)]);
    let mut state = 0x5555_6666u64;
    let source: TensorMap<_, num_complex::Complex32> =
        TensorMap::from_block_fn(&runtime(), [&left], [&right], move |_, _| {
            num_complex::Complex32::new(fill(&mut state) as f32, fill(&mut state) as f32)
        })
        .unwrap();
    svd_policy_sweep!(source, u1_leg(&[(-1, 1), (0, 2)]), "u1 c32");
}

#[test]
fn su2_svd_composition_matches_host_for_every_policy() {
    // SU(2): dim(c) = 2j+1, so the rank budget is quantum-dimension weighted
    // and cross-sector selection is not the naive value order.
    let leg = su2_leg(&[(0, 3), (1, 2), (2, 2)]);
    let mut state = 0x3333_4444u64;
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime(), [&leg], [&leg], move |_, _| fill(&mut state)).unwrap();
    svd_policy_sweep!(source, su2_leg(&[(0, 2), (2, 1)]), "su2 f64");
}

#[test]
fn su2_complex_svd_composition_matches_host_for_every_policy() {
    let leg = su2_leg(&[(0, 3), (1, 2), (2, 2)]);
    let mut state = 0x4444_5555u64;
    let source: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime(), [&leg], [&leg], move |_, _| {
            Complex64::new(fill(&mut state), fill(&mut state))
        })
        .unwrap();
    svd_policy_sweep!(source, su2_leg(&[(0, 2), (2, 1)]), "su2 c64");
}

#[test]
fn fermionic_svd_composition_matches_host_for_every_policy() {
    let leg = fz2_leg(&[(false, 3), (true, 3)]);
    let dual = leg.try_dual().unwrap();
    let mut state = 0x5555_6666u64;
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime(), [&leg, &dual], [&leg], move |_, _| {
            fill(&mut state)
        })
        .unwrap();
    svd_policy_sweep!(source, fz2_leg(&[(false, 2), (true, 1)]), "fz2 f64");
}

#[test]
fn fermionic_u1_product_svd_composition_matches_host_for_every_policy() {
    let provider = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let leg = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 3),
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(2)), 2),
        ],
    )
    .unwrap();
    let target = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 1),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
        ],
    )
    .unwrap();
    let mut state = 0x6666_7777u64;
    let source: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime(), [&leg], [&leg], move |_, _| {
            Complex64::new(fill(&mut state), fill(&mut state))
        })
        .unwrap();
    svd_policy_sweep!(source, target, "fz2xu1 c64");
}

#[test]
fn lazy_adjoint_svd_composition_matches_host() {
    let left = u1_leg(&[(0, 3), (1, 2)]);
    let right = u1_leg(&[(0, 2), (1, 3)]);
    let mut state = 0x7777_8888u64;
    let owned: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime(), [&left], [&right], move |_, _| fill(&mut state))
            .unwrap();
    let lazy = owned.adjoint().unwrap();
    svd_policy_sweep!(lazy, u1_leg(&[(0, 2), (1, 1)]), "u1 lazy adjoint");
}

// ---------------------------------------------------------------------------
// Hermitian eigendecomposition.
// ---------------------------------------------------------------------------

/// A Hermitian endomorphism with a mixed-sign spectrum, so `|lambda|` order is
/// not the signed order and the selection really is magnitude-driven.
fn hermitian_u1(seed: u64) -> TensorMap<U1FusionRule, f64> {
    let leg = u1_leg(&[(-1, 2), (0, 3), (1, 2)]);
    let mut state = seed;
    let raw: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime(), [&leg], [&leg], move |_, _| fill(&mut state)).unwrap();
    raw.add(&raw.adjoint().unwrap(), 1.0, 1.0).unwrap()
}

#[test]
fn u1_eigh_composition_matches_host_for_every_policy() {
    let source = hermitian_u1(0x8888_9999);
    eigh_policy_sweep!(source, u1_leg(&[(-1, 1), (0, 2)]), "u1 eigh f64");
}

#[test]
fn su2_eigh_composition_matches_host_for_every_policy() {
    let leg = su2_leg(&[(0, 3), (1, 2), (2, 2)]);
    let mut state = 0x9999_aaaau64;
    let raw: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime(), [&leg], [&leg], move |_, _| {
            Complex64::new(fill(&mut state), fill(&mut state))
        })
        .unwrap();
    let one = Complex64::new(1.0, 0.0);
    let source = raw.add(&raw.adjoint().unwrap(), one, one).unwrap();
    eigh_policy_sweep!(source, su2_leg(&[(0, 2), (2, 1)]), "su2 eigh c64");
}

#[test]
fn fermionic_eigh_composition_matches_host_for_every_policy() {
    let leg = fz2_leg(&[(false, 3), (true, 3)]);
    let mut state = 0xaaaa_bbbbu64;
    let raw: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime(), [&leg], [&leg], move |_, _| fill(&mut state)).unwrap();
    let source = raw.add(&raw.adjoint().unwrap(), 1.0, 1.0).unwrap();
    eigh_policy_sweep!(source, fz2_leg(&[(false, 2), (true, 1)]), "fz2 eigh f64");
}

// ---------------------------------------------------------------------------
// Decision shapes: whole sectors dropped, ties, discard-all, no-op.
// ---------------------------------------------------------------------------

#[test]
fn a_whole_sector_is_dropped_exactly_as_host_drops_it() {
    // Sector 1 carries a spectrum two orders of magnitude below sector 0, so a
    // relative cutoff removes it entirely rather than shortening it.
    let leg = u1_leg(&[(0, 3), (1, 2)]);
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime(), [&leg], [&leg], |trees, _| {
            if trees.coupled() == &U1Irrep::new(0) {
                1.0
            } else {
                1.0e-6
            }
        })
        .unwrap();
    let found = assert_svd_composition!(source, Truncation::relative_cutoff(1e-3).unwrap(), "drop");
    assert_eq!(
        found.selection.subspace().sectors().unwrap(),
        vec![U1Irrep::new(0)],
        "the low sector must be gone from the bond, not merely shortened"
    );
}

#[test]
fn within_sector_ties_at_the_cut_are_broken_as_host_breaks_them() {
    // A scalar multiple of an isometry: every singular value inside a sector is
    // exactly equal, so `Rank` has to cut inside a run of identical values.
    let leg = u1_leg(&[(0, 3), (1, 3)]);
    let source: TensorMap<_, f64> = TensorMap::id(&runtime(), [&leg]).unwrap().scale(2.5);
    assert_every_value_is_the_same_bit_pattern!(source.svd_compact().unwrap().1);
    for rank in [1usize, 2, 3, 4, 5] {
        assert_svd_composition!(source, Truncation::rank(rank), &format!("tie rank {rank}"));
    }
}

#[test]
fn cross_sector_exact_ties_are_broken_as_host_breaks_them() {
    // Two U(1) sectors with bit-identical spectra: every `Rank` cut that is not
    // a multiple of the sector count lands on an exact cross-sector tie.
    let leg = u1_leg(&[(0, 3), (1, 3), (2, 3)]);
    let source: TensorMap<_, f64> = TensorMap::id(&runtime(), [&leg]).unwrap();
    assert_every_value_is_the_same_bit_pattern!(source.svd_compact().unwrap().1);
    for rank in 0..=9usize {
        assert_svd_composition!(
            source,
            Truncation::rank(rank),
            &format!("cross tie rank {rank}")
        );
    }
}

#[test]
fn su2_cross_sector_exact_ties_are_broken_as_host_breaks_them() {
    // Same, with dim(c) != 1: the weighted budget overflows mid-tie.
    let leg = su2_leg(&[(0, 2), (1, 2), (2, 2)]);
    let source: TensorMap<_, f64> = TensorMap::id(&runtime(), [&leg]).unwrap();
    assert_every_value_is_the_same_bit_pattern!(source.svd_compact().unwrap().1);
    for rank in 0..=12usize {
        assert_svd_composition!(
            source,
            Truncation::rank(rank),
            &format!("su2 cross tie rank {rank}")
        );
    }
}

#[test]
fn signed_cross_sector_eigenvalue_ties_are_broken_as_host_breaks_them() {
    // `|lambda|` ties across sectors with *opposite signs*: the selection is
    // magnitude-driven, so +2 in one sector and -2 in another are an exact tie
    // that only the slice order can break, and the published `eigh_full` order
    // is by descending |lambda|, not by signed value.
    let leg = u1_leg(&[(0, 2), (1, 2), (2, 2)]);
    let runtime = runtime();
    let identity: TensorMap<_, f64> = TensorMap::id(&runtime, [&leg]).unwrap();
    // diag(+2, -2) in every sector: three sectors x two magnitudes, all equal.
    // Annotated: the payload dtype is taken solely from float literals, and
    // `f32` reaching `FactorizationScalar` (#1324) means `eigh_full` no longer
    // pins `{float}` to `f64` before `value.abs()` below resolves.
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] != indices[1] {
                0.0
            } else if indices[0] == 0 {
                2.0
            } else {
                -2.0
            }
        })
        .unwrap();
    assert_eq!(identity.block_count(), source.block_count());
    let magnitudes: Vec<f64> = source
        .eigh_full()
        .unwrap()
        .0
        .diagview()
        .unwrap()
        .iter()
        .flat_map(|entry| entry.values.iter().map(|value| value.abs()))
        .collect();
    assert!(
        magnitudes
            .iter()
            .all(|value| value.to_bits() == 2.0f64.to_bits()),
        "every |lambda| must be exactly 2.0 for this to be a tie test, got {magnitudes:?}"
    );
    for rank in 0..=6usize {
        assert_eigh_composition!(
            source,
            Truncation::rank(rank),
            &format!("signed cross tie rank {rank}")
        );
    }
}

#[test]
fn find_truncated_ignores_the_order_the_spectra_arrive_in() {
    // The documented contract: input order is irrelevant because the spectra
    // are ordered by `SectorId` before the decision. Reversing the input must
    // therefore change neither the subspace nor the error bits — and an exact
    // cross-sector tie is where it would show if the sort were missing.
    let leg = u1_leg(&[(0, 3), (1, 3), (2, 3)]);
    let source: TensorMap<_, f64> = TensorMap::id(&runtime(), [&leg]).unwrap();
    let (_, s, _) = source.svd_compact().unwrap();
    let bond = s.domain()[0].clone();
    let canonical = s.diagview().unwrap();
    let mut reversed = canonical.clone();
    reversed.reverse();
    assert_ne!(
        canonical.iter().map(|e| e.sector).collect::<Vec<_>>(),
        reversed.iter().map(|e| e.sector).collect::<Vec<_>>(),
        "the permuted input must actually differ"
    );
    for rank in 0..=9usize {
        let truncation = Truncation::rank(rank);
        let want = bond.find_truncated(&canonical, &truncation).unwrap();
        let got = bond.find_truncated(&reversed, &truncation).unwrap();
        assert_eq!(
            got.selection.subspace(),
            want.selection.subspace(),
            "rank {rank}"
        );
        assert_eq!(got.error.to_bits(), want.error.to_bits(), "rank {rank}");
        assert_eq!(
            s.restrict_diagonal(&got.selection).unwrap().data(),
            s.restrict_diagonal(&want.selection).unwrap().data(),
            "rank {rank}"
        );
    }
}

#[test]
fn dense_restrict_diagonal_equals_two_restrict_leg_calls() {
    // `restrict_diagonal`'s dense arm restricts both axes in one kernel call.
    // Its oracle is the already-merged single-axis primitive applied twice,
    // which shares no code path with the two-axis start table.
    let leg = u1_leg(&[(0, 4), (1, 3)]);
    let runtime = runtime();
    let mut state = 0x1357_9bdfu64;
    let dense: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], move |_, _| fill(&mut state)).unwrap();
    assert!(
        dense.diagonal_spectrum().unwrap().is_none(),
        "this fixture must exercise the dense arm"
    );
    for pairs in [
        vec![(U1Irrep::new(0), 0..2), (U1Irrep::new(1), 0..1)],
        // A non-prefix range: `find_truncated` never produces one, but
        // `restrict_diagonal` must not assume a leading prefix.
        vec![(U1Irrep::new(0), 1..4), (U1Irrep::new(1), 2..3)],
        // A sector dropped entirely.
        vec![(U1Irrep::new(1), 1..3)],
    ] {
        let selection = LegSelection::try_new(&leg, pairs.clone()).unwrap();
        let once = dense.restrict_diagonal(&selection).unwrap();
        let twice = dense
            .restrict_leg(0, &selection)
            .unwrap()
            .restrict_leg(1, &selection)
            .unwrap();
        assert_eq!(once.codomain(), twice.codomain(), "{pairs:?}");
        assert_eq!(once.domain(), twice.domain(), "{pairs:?}");
        assert_eq!(once.data(), twice.data(), "{pairs:?}");
    }
}

#[test]
fn discarding_everything_yields_the_empty_bond_and_host_s_empty_factors() {
    let leg = u1_leg(&[(0, 3), (1, 2)]);
    let mut state = 0xbbbb_ccccu64;
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime(), [&leg], [&leg], move |_, _| fill(&mut state)).unwrap();

    let (u, s, vh) = source.svd_compact().unwrap();
    let bond = s.domain()[0].clone();
    let found = bond
        .find_truncated(&s.diagview().unwrap(), &Truncation::rank(0))
        .unwrap();
    let selection = &found.selection;
    assert!(
        selection.subspace().sectors().unwrap().is_empty(),
        "discard-all must produce the empty bond leg, as Host svd_trunc does"
    );
    assert!(!selection.is_full(), "an empty selection of a nonempty leg");

    let host = source.svd_trunc(&Truncation::rank(0)).unwrap();
    let got_u = u.restrict_leg(u.codomain_rank(), selection).unwrap();
    let got_s = s.restrict_diagonal(selection).unwrap();
    let got_vh = vh.restrict_leg(0, selection).unwrap();
    assert_same_layout!(got_u, host.u, "discard-all u");
    assert_same_layout!(got_s, host.s, "discard-all s");
    assert_same_layout!(got_vh, host.vh, "discard-all vh");
    assert!(got_u.data().is_empty() && got_s.data().is_empty() && got_vh.data().is_empty());
    let spectra = s.diagview().unwrap();
    assert_truncation_error!(found, host.error, bond, spectra, "discard-all");

    // `embed_leg` with the empty selection: the adjoint of restricting to
    // nothing is the zero map back onto the parent leg.
    let embedded = got_u.embed_leg(got_u.codomain_rank(), selection).unwrap();
    assert_eq!(embedded.domain()[0], bond);
    assert!(embedded.data().iter().all(|&value| value == 0.0));
}

#[test]
fn a_no_op_decision_is_reported_as_full_and_copies_the_same_bits() {
    let leg = u1_leg(&[(0, 3), (1, 2)]);
    let mut state = 0xcccc_ddddu64;
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime(), [&leg], [&leg], move |_, _| fill(&mut state)).unwrap();
    let (u, s, _) = source.svd_compact().unwrap();
    let bond = s.domain()[0].clone();
    let found = bond
        .find_truncated(&s.diagview().unwrap(), &Truncation::Full)
        .unwrap();
    assert!(found.selection.is_full());
    assert_eq!(found.error.to_bits(), 0.0_f64.to_bits());
    assert_eq!(
        u.restrict_leg(u.codomain_rank(), &found.selection)
            .unwrap()
            .data(),
        u.data(),
        "a full selection copies the payload unchanged"
    );
}

#[test]
fn is_full_holds_for_the_only_selection_of_an_empty_leg() {
    let empty = u1_leg(&[]);
    let found = empty
        .find_truncated(&[] as &[SectorSpectrum<U1Irrep, f64>], &Truncation::rank(5))
        .unwrap();
    assert!(found.selection.is_full());
    assert_eq!(found.error.to_bits(), 0.0_f64.to_bits());
}

// ---------------------------------------------------------------------------
// diagview
// ---------------------------------------------------------------------------

#[test]
fn diagview_reads_the_same_values_from_compact_and_dense_storage() {
    let leg = u1_leg(&[(0, 3), (1, 2)]);
    let runtime = runtime();
    let mut state = 0xdddd_eeeeu64;
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], move |_, _| fill(&mut state)).unwrap();
    let (_, s, _) = source.svd_compact().unwrap();
    assert!(s.diagonal_spectrum().unwrap().is_some(), "s is compact");
    let compact = s.diagview().unwrap();

    // The same map, forced through dense storage. `add` of a compact factor and
    // a dense zero produces the dense twin (`diagonal_spectrum` is then `None`,
    // which is the contract `diagview` deliberately does not change).
    let zero: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&leg], [&leg]).unwrap();
    let dense = zero.add(&s, 1.0, 1.0).unwrap();
    assert!(
        dense.diagonal_spectrum().unwrap().is_none(),
        "the dense twin must still report no compact storage"
    );
    assert_eq!(dense.diagview().unwrap(), compact);

    // Off-diagonal entries are not inspected: a dense map with a nonzero
    // off-diagonal still reports its diagonal.
    let full: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                (indices[0] + 1) as f64
            } else {
                7.0
            }
        })
        .unwrap();
    let read = full.diagview().unwrap();
    assert_eq!(read.len(), 2);
    for entry in &read {
        let expected: Vec<f64> = (0..entry.values.len()).map(|i| (i + 1) as f64).collect();
        assert_eq!(entry.values, expected);
    }
}

#[test]
fn diagview_rejects_a_non_bond_map_and_a_lazy_adjoint() {
    let leg = u1_leg(&[(0, 2), (1, 2)]);
    let other = u1_leg(&[(0, 3), (1, 1)]);
    let runtime = runtime();

    let rank_three: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&leg, &leg], [&leg]).unwrap();
    assert!(matches!(
        rank_three.diagview(),
        Err(Error::InvalidArgument(message)) if message.contains("rank-(1,1)")
    ));

    let rectangular: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&leg], [&other]).unwrap();
    assert!(matches!(
        rectangular.diagview(),
        Err(Error::InvalidArgument(message)) if message.contains("equal codomain and domain legs")
    ));

    let square: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&leg], [&leg]).unwrap();
    assert!(matches!(
        square.adjoint().unwrap().diagview(),
        Err(Error::InvalidArgument(message)) if message.contains("lazy adjoint")
    ));
}

// ---------------------------------------------------------------------------
// restrict_diagonal preconditions
// ---------------------------------------------------------------------------

#[test]
fn restrict_diagonal_rejects_everything_it_documents() {
    let leg = u1_leg(&[(0, 3), (1, 2)]);
    let runtime = runtime();
    let selection = LegSelection::try_new(&leg, [(U1Irrep::new(0), 0..2)]).unwrap();

    let rank_three: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&leg, &leg], [&leg]).unwrap();
    assert!(matches!(
        rank_three.restrict_diagonal(&selection),
        Err(Error::InvalidArgument(message)) if message.contains("rank-(1,1)")
    ));

    let other = u1_leg(&[(0, 3), (1, 1)]);
    let rectangular: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&leg], [&other]).unwrap();
    assert!(matches!(
        rectangular.restrict_diagonal(&selection),
        Err(Error::InvalidArgument(message)) if message.contains("equal codomain and domain legs")
    ));

    let foreign: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&other], [&other]).unwrap();
    assert!(matches!(
        foreign.restrict_diagonal(&selection),
        Err(Error::InvalidArgument(message))
            if message.contains("not the leg this selection was built from")
    ));

    let square: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&leg], [&leg]).unwrap();
    assert!(matches!(
        square.adjoint().unwrap().restrict_diagonal(&selection),
        Err(Error::InvalidArgument(message)) if message.contains("lazy adjoint")
    ));

    // A selection from a different *rule* cannot be spelled: `LegSelection<R>`
    // is typed by the provider, so `restrict_diagonal`'s `RuleMismatch` guard
    // only fires for two providers of the same type with different identities,
    // which is `restrict_leg`'s already-tested case.
}

#[test]
fn restrict_diagonal_keeps_a_compact_payload_compact() {
    let leg = u1_leg(&[(0, 4), (1, 3)]);
    let mut state = 0xeeee_ffffu64;
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime(), [&leg], [&leg], move |_, _| fill(&mut state)).unwrap();
    let (_, s, _) = source.svd_compact().unwrap();
    let bond = s.domain()[0].clone();
    let selection = LegSelection::try_new(&bond, [(U1Irrep::new(0), 0..2)]).unwrap();
    let restricted = s.restrict_diagonal(&selection).unwrap();
    let spectrum = restricted
        .diagonal_spectrum()
        .unwrap()
        .expect("a compact input must stay compact");
    assert_eq!(spectrum.len(), 1);
    assert_eq!(spectrum[0].values.len(), 2);
    assert_eq!(
        spectrum[0].values,
        s.diagview().unwrap()[0].values[0..2].to_vec()
    );
}

// ---------------------------------------------------------------------------
// find_truncated input validation
// ---------------------------------------------------------------------------

fn spectrum(charge: i32, values: &[f64]) -> SectorSpectrum<U1Irrep, f64> {
    SectorSpectrum {
        sector: U1Irrep::new(charge),
        values: values.to_vec(),
    }
}

#[test]
fn find_truncated_rejects_a_malformed_input_spectrum() {
    let leg = u1_leg(&[(0, 3), (1, 2)]);
    let full = Truncation::Full;

    let missing = [spectrum(0, &[3.0, 2.0, 1.0])];
    assert!(matches!(
        leg.find_truncated(&missing, &full),
        Err(Error::InvalidArgument(message)) if message.contains("one spectrum per leg sector")
    ));

    let duplicated = [spectrum(0, &[3.0, 2.0, 1.0]), spectrum(0, &[3.0, 2.0, 1.0])];
    assert!(matches!(
        leg.find_truncated(&duplicated, &full),
        Err(Error::InvalidArgument(message)) if message.contains("more than one spectrum")
    ));

    let unknown = [spectrum(0, &[3.0, 2.0, 1.0]), spectrum(7, &[2.0, 1.0])];
    assert!(matches!(
        leg.find_truncated(&unknown, &full),
        Err(Error::InvalidArgument(message)) if message.contains("absent from the truncated leg")
    ));

    for wrong in [&[3.0, 2.0][..], &[3.0, 2.0, 1.0, 0.5][..]] {
        let bad = [spectrum(0, wrong), spectrum(1, &[2.0, 1.0])];
        assert!(matches!(
            leg.find_truncated(&bad, &full),
            Err(Error::InvalidArgument(message)) if message.contains("expected the leg degeneracy")
        ));
    }

    for values in [
        &[1.0, 2.0, 3.0][..],
        &[f64::NAN, 2.0, 1.0][..],
        &[f64::INFINITY, 2.0, 1.0][..],
    ] {
        let bad = [spectrum(0, values), spectrum(1, &[2.0, 1.0])];
        assert!(
            leg.find_truncated(&bad, &full).is_err(),
            "ascending, NaN and infinite spectra must be rejected: {values:?}"
        );
    }
}

#[test]
fn find_truncated_rejects_a_truncation_space_from_another_rule() {
    let leg = u1_leg(&[(0, 3), (1, 2)]);
    let foreign = su2_leg(&[(0, 1)]);
    let spectra = [spectrum(0, &[3.0, 2.0, 1.0]), spectrum(1, &[2.0, 1.0])];
    assert!(leg
        .find_truncated(&spectra, &Truncation::space(foreign.truncspace()))
        .is_err());
}

#[test]
fn a_selection_from_another_leg_is_rejected_by_both_appliers() {
    let leg = u1_leg(&[(0, 3), (1, 2)]);
    let other = u1_leg(&[(0, 4), (1, 2)]);
    let runtime = runtime();
    let selection = LegSelection::try_new(&other, [(U1Irrep::new(0), 0..2)]).unwrap();
    let tensor: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&leg], [&leg]).unwrap();
    assert!(tensor.restrict_diagonal(&selection).is_err());
    assert!(tensor.restrict_leg(0, &selection).is_err());
}

/// A caller generic over the payload can name every bound `find_truncated`
/// needs: `SpectrumMagnitude` is re-exported from `tenet::typed`, so the
/// device recipe does not have to map the spectrum to `f64` magnitudes first
/// (#1300 left it unnameable outside the crate; #1297 exports it).
fn find_truncated_generically<R, D>(
    s: &TensorMap<R, D>,
    truncation: &Truncation,
) -> TruncatedSelection<R>
where
    R: tenet::core::MultiplicityFreeRigidSymbols<Scalar = f64>
        + tenet::core::CheckedFusionAlgebra
        + tenet::typed::SectorCodec,
    D: tenet::prelude::TensorScalar + SpectrumMagnitude,
{
    s.domain()[0]
        .find_truncated(&s.diagview().unwrap(), truncation)
        .unwrap()
}

#[test]
fn a_payload_generic_caller_can_name_the_find_truncated_bound() {
    let runtime = runtime();
    let leg = u1_leg(&[(0, 3), (1, 2)]);
    let truncation = Truncation::rank(2);

    let mut state = 7;
    let real: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| fill(&mut state)).unwrap();
    let (_, s, _) = real.svd_compact().unwrap();
    let found = find_truncated_generically(&s, &truncation);
    let host = real.svd_trunc(&truncation).unwrap();
    assert_eq!(
        found.selection.subspace(),
        &host.s.domain()[0],
        "f64 kept bond"
    );
    assert_truncation_error!(found, host.error, leg, s.diagview().unwrap(), "f64");

    let complex: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| {
            Complex64::new(fill(&mut state), fill(&mut state))
        })
        .unwrap();
    let (_, s, _) = complex.svd_compact().unwrap();
    let found = find_truncated_generically(&s, &truncation);
    let host = complex.svd_trunc(&truncation).unwrap();
    assert_eq!(
        found.selection.subspace(),
        &host.s.domain()[0],
        "c64 kept bond"
    );
    assert_truncation_error!(found, host.error, leg, s.diagview().unwrap(), "c64");
}
