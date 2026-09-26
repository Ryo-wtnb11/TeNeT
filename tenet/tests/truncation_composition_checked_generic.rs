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
//! The gate is the multiplicity-free one (`truncation_composition.rs`): the
//! kept bond and every factor's spaces and block geometry exactly, then the
//! kept values, the gauge-free factor relations and the truncation error
//! under the workspace tolerance rule (`docs/testing_numerics.md`), all
//! against the TeNeT-independent expectation of `truncation_oracle`.

#![cfg(feature = "racah-generated")]

use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use tenet::prelude::{Runtime, TensorMap};
use tenet::typed::{Eig, Eigh, GradedSpace, SUNFusionRule, Svd};

#[path = "../../tests/support/numerics.rs"]
mod numerics;
#[macro_use]
mod truncation_oracle;

use truncation_oracle::{
    assert_error_close, assert_kept_magnitudes, discarded_norm, select, triangular_eigenvalues,
    Offer,
};

fn runtime() -> Runtime {
    Runtime::builder().dense_threads(1).build().unwrap()
}

fn fill(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as f64) / (u32::MAX as f64) - 0.5
}

/// Weyl dimension of the SU(3) irrep with Dynkin labels `(p, q)`:
/// `(p + 1)(q + 1)(p + q + 2) / 2`.
fn su3_dim(labels: &[i64]) -> f64 {
    let (p, q) = (labels[0] as f64, labels[1] as f64);
    (p + 1.0) * (q + 1.0) * (p + q + 2.0) / 2.0
}

/// `x^H` as an owned tensor: checked-Generic `compose` takes owned operands
/// only, and `add` is the owned copy that accepts a lazy adjoint.
macro_rules! owned_adjoint {
    ($x:expr, $one:expr) => {{
        let adjoint = $x.adjoint().unwrap();
        adjoint.axpby($one, &adjoint, $one - $one).unwrap()
    }};
}

/// `lhs` and `rhs` are `compose` results on one homspace and agree within the
/// tolerance rule.
macro_rules! assert_relation {
    ($lhs:expr, $rhs:expr, $terms:expr, $what:expr) => {{
        let (lhs, rhs) = (&$lhs, &$rhs);
        assert_eq!(lhs.codomain(), rhs.codomain(), "{} codomain", $what);
        assert_eq!(lhs.domain(), rhs.domain(), "{} domain", $what);
        numerics::assert_slices_close(&$what, lhs.data(), rhs.data(), $terms);
    }};
}

/// `x^H x = 1` on the bond. `TensorMap::id` is multiplicity-free only, so the
/// identity is spelled by its reduced blocks.
macro_rules! assert_isometry {
    ($gram:expr, $bond:expr, $one:expr, $terms:expr, $what:expr) => {{
        let gram = $gram;
        let (one, zero) = ($one, $one - $one);
        let identity = TensorMap::from_block_fn(gram.runtime(), [&$bond], [&$bond], |_, index| {
            if index[0] == index[1] {
                one
            } else {
                zero
            }
        })
        .unwrap();
        assert_relation!(gram, identity, $terms, $what);
    }};
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

macro_rules! assert_su3_svd_composition {
    ($source:expr, $one:expr, $target:expr, $tag:expr) => {{
        let source = $source;
        let offers = singular_offers!(source, su3_dim);
        let terms = source.data().len();
        for (name, truncation, policy) in policies!($target) {
            let case = format!("{} {name}", $tag);
            let Svd { u, s, vh } = source.svd_compact().unwrap();
            assert!(
                s.diagonal_spectrum().unwrap().is_none(),
                "checked-Generic compact s is dense, which is what exercises diagview's strided arm"
            );
            let bond = s.domain()[0].clone();
            let found = bond.find_truncated(&s.diagview().unwrap(), &truncation).unwrap();
            let selection = &found.selection;
            let got_u = u.restrict_leg(u.codomain_rank(), selection).unwrap();
            let got_s = s.restrict_diagonal(selection).unwrap();
            let got_vh = vh.restrict_leg(0, selection).unwrap();
            let kept = select(&offers, &policy);

            let kept_bond = got_s.domain()[0].clone();
            assert_kept_bond!(kept_bond, offers, kept, case);
            assert_eq!(got_s.codomain(), got_s.domain(), "{case}: s is a bond map");
            assert_eq!(got_u.codomain(), source.codomain(), "{case}: u codomain");
            assert_eq!(got_u.domain(), got_s.domain(), "{case}: u domain");
            assert_eq!(got_vh.codomain(), got_s.domain(), "{case}: vh codomain");
            assert_eq!(got_vh.domain(), source.domain(), "{case}: vh domain");
            assert_canonical_layout!(got_u, format!("{case}: u"));
            assert_canonical_layout!(got_s, format!("{case}: s"));
            assert_canonical_layout!(got_vh, format!("{case}: vh"));

            assert_kept_magnitudes(&case, &got_s.diagview().unwrap(), &offers, terms);
            assert_relation!(
                source.compose(&owned_adjoint!(got_vh, $one)).unwrap(),
                got_u.compose(&got_s).unwrap(),
                terms,
                format!("{case}: t vh^H = u s")
            );
            assert_relation!(
                owned_adjoint!(got_u, $one).compose(&source).unwrap(),
                got_s.compose(&got_vh).unwrap(),
                terms,
                format!("{case}: u^H t = s vh")
            );
            assert_isometry!(
                owned_adjoint!(got_u, $one).compose(&got_u).unwrap(),
                kept_bond,
                $one,
                terms,
                format!("{case}: u^H u")
            );
            assert_isometry!(
                got_vh.compose(&owned_adjoint!(got_vh, $one)).unwrap(),
                kept_bond,
                $one,
                terms,
                format!("{case}: vh vh^H")
            );
            assert_error_close(
                &case,
                source.data(),
                found.error,
                discarded_norm(&offers, &kept),
                terms,
            );
        }
    }};
}

macro_rules! assert_su3_eigh_composition {
    ($source:expr, $one:expr, $target:expr, $tag:expr) => {{
        let source = $source;
        // A Hermitian block's singular values are its |lambda|.
        let offers = singular_offers!(source, su3_dim);
        let terms = source.data().len();
        for (name, truncation, policy) in policies!($target) {
            let case = format!("{} {name}", $tag);
            let Eigh { d, v } = source.eigh_full().unwrap();
            let bond = d.domain()[0].clone();
            let found = bond
                .find_truncated(&d.diagview().unwrap(), &truncation)
                .unwrap();
            let selection = &found.selection;
            let got_d = d.restrict_diagonal(selection).unwrap();
            let got_v = v.restrict_leg(v.codomain_rank(), selection).unwrap();
            let kept = select(&offers, &policy);

            let kept_bond = got_d.domain()[0].clone();
            assert_kept_bond!(kept_bond, offers, kept, case);
            assert_eq!(got_d.codomain(), got_d.domain(), "{case}: d is a bond map");
            assert_eq!(got_v.codomain(), source.codomain(), "{case}: v codomain");
            assert_eq!(got_v.domain(), got_d.domain(), "{case}: v domain");
            assert_canonical_layout!(got_d, format!("{case}: d"));
            assert_canonical_layout!(got_v, format!("{case}: v"));

            assert_kept_magnitudes(&case, &got_d.diagview().unwrap(), &offers, terms);
            assert_relation!(
                source.compose(&got_v).unwrap(),
                got_v.compose(&got_d).unwrap(),
                terms,
                format!("{case}: t v = v d")
            );
            assert_isometry!(
                owned_adjoint!(got_v, $one).compose(&got_v).unwrap(),
                kept_bond,
                $one,
                terms,
                format!("{case}: v^H v")
            );
            assert_error_close(
                &case,
                source.data(),
                found.error,
                discarded_norm(&offers, &kept),
                terms,
            );
        }
    }};
}

/// The general eigendecomposition on a triangular source, whose eigenvalues
/// are its diagonal. `$complex` is the source lifted to the factor dtype.
macro_rules! assert_su3_eig_composition {
    ($source:expr, $complex:expr, $target:expr, $tag:expr) => {{
        let source = $source;
        let complex = $complex;
        let references: Vec<(Vec<i64>, Vec<Complex64>)> = sector_matrices!(source)
            .iter()
            .map(|(sector, matrix)| (sector.clone(), triangular_eigenvalues(matrix)))
            .collect();
        let offers: Vec<Offer<Vec<i64>>> = references
            .iter()
            .map(|(sector, values)| Offer {
                sector: sector.clone(),
                dim: su3_dim(sector),
                magnitudes: values.iter().map(|value| value.norm()).collect(),
            })
            .collect();
        let terms = source.data().len();
        for (name, truncation, policy) in policies!($target) {
            let case = format!("{} {name}", $tag);
            let Eig { d, v } = source.eig_full().unwrap();
            let bond = d.domain()[0].clone();
            let found = bond
                .find_truncated(&d.diagview().unwrap(), &truncation)
                .unwrap();
            let selection = &found.selection;
            let got_d = d.restrict_diagonal(selection).unwrap();
            let got_v = v.restrict_leg(v.codomain_rank(), selection).unwrap();
            let kept = select(&offers, &policy);

            assert_kept_bond!(got_d.domain()[0], offers, kept, case);
            assert_eq!(got_d.codomain(), got_d.domain(), "{case}: d is a bond map");
            assert_eq!(got_v.codomain(), source.codomain(), "{case}: v codomain");
            assert_eq!(got_v.domain(), got_d.domain(), "{case}: v domain");
            assert_canonical_layout!(got_d, format!("{case}: d"));
            assert_canonical_layout!(got_v, format!("{case}: v"));
            for entry in got_d.diagview().unwrap() {
                let (_, reference) = references
                    .iter()
                    .find(|(sector, _)| *sector == entry.sector)
                    .unwrap();
                numerics::assert_slices_close(
                    &format!("{case}: kept eigenvalues of {:?}", entry.sector),
                    &entry.values,
                    &reference[..entry.values.len()],
                    terms,
                );
            }
            assert_relation!(
                complex.compose(&got_v).unwrap(),
                got_v.compose(&got_d).unwrap(),
                terms,
                format!("{case}: t v = v d")
            );
            assert_error_close(
                &case,
                source.data(),
                found.error,
                discarded_norm(&offers, &kept),
                terms,
            );
        }
    }};
}

#[test]
fn su3_svd_composition_matches_the_oracle_for_every_policy() {
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
    assert_su3_svd_composition!(source, 1.0, su3_target(&provider), "su3 f64");
}

#[test]
fn su3_complex_svd_composition_matches_the_oracle_for_every_policy() {
    let (provider, leg) = su3_legs();
    let mut state = 0x5150_2703u64;
    let source: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime(), [&leg, &leg], [&leg], move |_, _| {
            Complex64::new(fill(&mut state), fill(&mut state))
        })
        .unwrap();
    assert_su3_svd_composition!(source, Complex64::ONE, su3_target(&provider), "su3 c64");
}

#[test]
fn su3_eigh_composition_matches_the_oracle_for_every_policy() {
    let (provider, leg) = su3_legs();
    let mut state = 0x5150_2702u64;
    let raw: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime(), [&leg], [&leg], move |_, _| fill(&mut state)).unwrap();
    let source = raw.axpby(1.0, &raw.adjoint().unwrap(), 1.0).unwrap();
    assert_su3_eigh_composition!(source, 1.0, su3_target(&provider), "su3 eigh f64");
}

#[test]
fn su3_complex_eigh_composition_matches_the_oracle_for_every_policy() {
    let (provider, leg) = su3_legs();
    let mut state = 0x5150_2704u64;
    let raw: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime(), [&leg], [&leg], move |_, _| {
            Complex64::new(fill(&mut state), fill(&mut state))
        })
        .unwrap();
    let one = Complex64::new(1.0, 0.0);
    let source = raw.axpby(one, &raw.adjoint().unwrap(), one).unwrap();
    assert_su3_eigh_composition!(
        source,
        Complex64::ONE,
        su3_target(&provider),
        "su3 eigh c64"
    );
}

#[test]
fn su3_eig_composition_matches_the_oracle_for_every_policy() {
    let (provider, leg) = su3_legs();
    let mut state = 0x5150_2705u64;
    let source: TensorMap<_, f64> = TensorMap::from_block_fn(
        &runtime(),
        [&leg],
        [&leg],
        move |_, indices| match indices[0].cmp(&indices[1]) {
            std::cmp::Ordering::Equal => 4.0 * fill(&mut state),
            std::cmp::Ordering::Less => 0.5 * fill(&mut state),
            std::cmp::Ordering::Greater => 0.0,
        },
    )
    .unwrap();
    let complex = source.to_c64();
    assert_su3_eig_composition!(source, complex, su3_target(&provider), "su3 eig f64");
}
