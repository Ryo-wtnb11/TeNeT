//! The truncation composition on a checked Generic provider (#1300).
//!
//! SU(3) with the adjoint irrep is the case the multiplicity-free file cannot
//! reach: the quantum-dimension weight is the fallible `try_sqrt_dim_scalar²`,
//! and the compact SVD's `s` is **dense**, so `diagview` takes its strided arm
//! rather than cloning a stored spectrum.
//!
//! The SVD source is rank `(2,1)` on purpose. A `bond <- bond` map has
//! rank-one trees and therefore no outer-multiplicity vertex at all; with two
//! adjoint legs in the codomain, `8 ⊗ 8 -> 8` carries two fusion channels, so
//! `u`'s blocks differ only by their vertex label and the restriction has to
//! leave that part of the key alone.
//!
//! The gate is the same as the multiplicity-free one: spaces, block geometry
//! and the kept bond exactly, then payloads and the truncation error under the
//! workspace tolerance rule (`docs/testing_numerics.md`), the error also
//! against an independent discarded-norm oracle.

#![cfg(feature = "racah-generated")]

use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use tenet::prelude::{Runtime, TensorMap, Truncation};
use tenet::typed::{GradedSpace, SUNFusionRule, SpectrumMagnitude};

#[path = "../../tests/support/numerics.rs"]
mod numerics;

fn runtime() -> Runtime {
    Runtime::builder().dense_threads(1).build().unwrap()
}

fn fill(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as f64) / (u32::MAX as f64) - 0.5
}

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

/// Weyl dimension of the SU(3) irrep with Dynkin labels `(p, q)`:
/// `(p + 1)(q + 1)(p + q + 2) / 2`.
fn su3_dim(labels: &[i64]) -> f64 {
    let (p, q) = (labels[0] as f64, labels[1] as f64);
    (p + 1.0) * (q + 1.0) * (p + q + 2.0) / 2.0
}

/// Checks the error against Host and against `sqrt(sum_c dim(c) sum_discarded
/// |v|^2)` over the offered spectrum past each sector's kept prefix; `terms`
/// is the number of offered values.
fn assert_truncation_error<V: SpectrumMagnitude>(
    found: &tenet::typed::TruncatedSelection<SUNFusionRule>,
    host_error: f64,
    spectra: &[tenet::typed::SectorSpectrum<Vec<i64>, V>],
    case: &str,
) {
    let kept = found.selection.subspace();
    let kept_sectors = kept.sectors().unwrap();
    let mut sum = 0.0f64;
    for entry in spectra {
        let prefix = kept_sectors
            .iter()
            .position(|sector| *sector == entry.sector)
            .map_or(0, |index| kept.degeneracies()[index]);
        for &value in &entry.values[prefix..] {
            let magnitude = value.magnitude();
            sum += su3_dim(&entry.sector) * magnitude * magnitude;
        }
    }
    let terms: usize = spectra.iter().map(|e| e.values.len()).sum();
    numerics::assert_close(
        &format!("{case}: truncation error against Host"),
        found.error,
        host_error,
        terms,
    );
    numerics::assert_close(
        &format!("{case}: truncation error against the discarded-norm oracle"),
        found.error,
        sum.sqrt(),
        terms,
    );
}

fn su3_legs() -> (Arc<SUNFusionRule>, GradedSpace<SUNFusionRule>) {
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [(vec![1i64, 0], 3), (vec![2i64, 2], 2)],
    )
    .unwrap();
    (provider, leg)
}

/// A smaller space on the same provider, used as a `truncspace` target.
fn su3_target(provider: &Arc<SUNFusionRule>) -> GradedSpace<SUNFusionRule> {
    GradedSpace::try_new_with_arc(Arc::clone(provider), [(vec![2i64, 2], 2)]).unwrap()
}

fn policies(target: &GradedSpace<SUNFusionRule>) -> Vec<(&'static str, Truncation)> {
    vec![
        ("Full", Truncation::Full),
        ("Rank(4)", Truncation::rank(4)),
        ("Rank(1)", Truncation::rank(1)),
        ("Rank(0)", Truncation::rank(0)),
        ("Tolerance", Truncation::relative_cutoff(0.25).unwrap()),
        (
            "ToleranceInf",
            Truncation::relative_inf_cutoff(0.4).unwrap(),
        ),
        ("DiscardWeight", Truncation::relative_error(0.2).unwrap()),
        ("Space", Truncation::space(target.truncspace())),
        (
            "Space & Rank",
            Truncation::space(target.truncspace()).and(Truncation::rank(2)),
        ),
    ]
}

macro_rules! assert_su3_svd_composition {
    ($source:expr, $target:expr, $tag:expr) => {{
        let source = $source;
        for (name, truncation) in policies(&$target) {
            let case = format!("{} {name}", $tag);
            let (u, s, vh) = source.svd_compact().unwrap();
            assert!(
                s.diagonal_spectrum().unwrap().is_none(),
                "checked-Generic compact s is dense, which is what exercises diagview's strided arm"
            );
            let bond = s.domain()[0].clone();
            let spectra = s.diagview().unwrap();
            let found = bond.find_truncated(&spectra, &truncation).unwrap();
            let selection = &found.selection;
            let host = source.svd_trunc(&truncation).unwrap();

            let got_u = u.restrict_leg(u.codomain_rank(), selection).unwrap();
            let got_s = s.restrict_diagonal(selection).unwrap();
            let got_vh = vh.restrict_leg(0, selection).unwrap();

            assert_eq!(*selection.subspace(), host.s.domain()[0], "{case}: bond");
            // Both routes factorize the same input and then only copy; every
            // source entry of a block can reach every factor entry of it.
            let terms = source.data().len();
            assert_same_layout!(got_u, host.u, format!("{case}: u"));
            assert_same_layout!(got_s, host.s, format!("{case}: s"));
            assert_same_layout!(got_vh, host.vh, format!("{case}: vh"));
            numerics::assert_slices_close(
                &format!("{case}: u payload"),
                got_u.data(),
                host.u.data(),
                terms,
            );
            numerics::assert_slices_close(
                &format!("{case}: s payload"),
                got_s.data(),
                host.s.data(),
                terms,
            );
            numerics::assert_slices_close(
                &format!("{case}: vh payload"),
                got_vh.data(),
                host.vh.data(),
                terms,
            );
            assert_truncation_error(&found, host.error, &spectra, &case);
        }
    }};
}

macro_rules! assert_su3_eigh_composition {
    ($source:expr, $target:expr, $tag:expr) => {{
        let source = $source;
        for (name, truncation) in policies(&$target) {
            let case = format!("{} {name}", $tag);
            let (d, v) = source.eigh_full().unwrap();
            let bond = d.domain()[0].clone();
            let spectra = d.diagview().unwrap();
            let found = bond.find_truncated(&spectra, &truncation).unwrap();
            let selection = &found.selection;
            let host = source.eigh_trunc(&truncation).unwrap();

            let got_d = d.restrict_diagonal(selection).unwrap();
            let got_v = v.restrict_leg(v.codomain_rank(), selection).unwrap();

            assert_eq!(*selection.subspace(), host.d.domain()[0], "{case}: bond");
            // Both routes factorize the same input and then only copy.
            let terms = source.data().len();
            assert_same_layout!(got_d, host.d, format!("{case}: d"));
            assert_same_layout!(got_v, host.v, format!("{case}: v"));
            numerics::assert_slices_close(
                &format!("{case}: d payload"),
                got_d.data(),
                host.d.data(),
                terms,
            );
            numerics::assert_slices_close(
                &format!("{case}: v payload"),
                got_v.data(),
                host.v.data(),
                terms,
            );
            assert_truncation_error(&found, host.error, &spectra, &case);
        }
    }};
}

#[test]
fn su3_svd_composition_matches_host_for_every_policy() {
    let (provider, leg) = su3_legs();
    let mut state = 0x5150_2701u64;
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime(), [&leg, &leg], [&leg], move |_, _| {
            fill(&mut state)
        })
        .unwrap();
    assert!(
        (0..source.block_count()).any(|index| source
            .block_fusion_trees(index)
            .unwrap()
            .codomain_vertices()
            .iter()
            .any(|vertex| vertex.get() == 2)),
        "the fixture must carry a Generic outer-multiplicity vertex mu = 2"
    );
    assert_su3_svd_composition!(source, su3_target(&provider), "su3 f64");
}

#[test]
fn su3_complex_svd_composition_matches_host_for_every_policy() {
    let (provider, leg) = su3_legs();
    let mut state = 0x5150_2703u64;
    let source: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime(), [&leg, &leg], [&leg], move |_, _| {
            Complex64::new(fill(&mut state), fill(&mut state))
        })
        .unwrap();
    assert_su3_svd_composition!(source, su3_target(&provider), "su3 c64");
}

#[test]
fn su3_eigh_composition_matches_host_for_every_policy() {
    let (provider, leg) = su3_legs();
    let mut state = 0x5150_2702u64;
    let raw: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime(), [&leg], [&leg], move |_, _| fill(&mut state)).unwrap();
    let source = raw.add(&raw.adjoint().unwrap(), 1.0, 1.0).unwrap();
    assert_su3_eigh_composition!(source, su3_target(&provider), "su3 eigh f64");
}

#[test]
fn su3_complex_eigh_composition_matches_host_for_every_policy() {
    let (provider, leg) = su3_legs();
    let mut state = 0x5150_2704u64;
    let raw: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime(), [&leg], [&leg], move |_, _| {
            Complex64::new(fill(&mut state), fill(&mut state))
        })
        .unwrap();
    let one = Complex64::new(1.0, 0.0);
    let source = raw.add(&raw.adjoint().unwrap(), one, one).unwrap();
    assert_su3_eigh_composition!(source, su3_target(&provider), "su3 eigh c64");
}
