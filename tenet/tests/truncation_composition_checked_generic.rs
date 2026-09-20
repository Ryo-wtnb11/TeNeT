//! The truncation composition on a checked Generic provider (#1300).
//!
//! SU(3) with the adjoint irrep is the case the multiplicity-free file cannot
//! reach: the quantum-dimension weight is the fallible `try_sqrt_dim_scalar²`,
//! the compact SVD's `s` is **dense**, so `diagview` takes its strided arm, and
//! the blocks carry outer-multiplicity vertices that a degeneracy-only
//! restriction must leave alone.
//!
//! The gate is the same as the multiplicity-free one: spaces and block
//! geometry first, then payload bits, then the truncation error.

#![cfg(feature = "racah-generated")]

use std::sync::Arc;

use tenet::prelude::{Runtime, TensorMap, Truncation};
use tenet::typed::{GradedSpace, SUNFusionRule};

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

fn su3_endomorphism(seed: u64) -> (Arc<SUNFusionRule>, TensorMap<SUNFusionRule, f64>) {
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [(vec![1i64, 0], 3), (vec![2i64, 2], 2)],
    )
    .unwrap();
    let mut state = seed;
    let tensor =
        TensorMap::from_block_fn(&runtime(), [&leg], [&leg], move |_, _| fill(&mut state)).unwrap();
    (provider, tensor)
}

fn policies() -> Vec<(&'static str, Truncation)> {
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
    ]
}

#[test]
fn su3_svd_composition_matches_host_for_every_policy() {
    let (_, source) = su3_endomorphism(0x5150_2701);
    for (name, truncation) in policies() {
        let (u, s, vh) = source.svd_compact().unwrap();
        assert!(
            s.diagonal_spectrum().unwrap().is_none(),
            "checked-Generic compact s is dense, which is what exercises diagview's strided arm"
        );
        let bond = s.domain()[0].clone();
        let found = bond
            .find_truncated(&s.diagview().unwrap(), &truncation)
            .unwrap();
        let selection = &found.selection;
        let host = source.svd_trunc(&truncation).unwrap();

        let got_u = u.restrict_leg(u.codomain_rank(), selection).unwrap();
        let got_s = s.restrict_diagonal(selection).unwrap();
        let got_vh = vh.restrict_leg(0, selection).unwrap();

        assert_eq!(*selection.subspace(), host.s.domain()[0], "{name}: bond");
        assert_same_layout!(got_u, host.u, format!("su3 {name}: u"));
        assert_same_layout!(got_s, host.s, format!("su3 {name}: s"));
        assert_same_layout!(got_vh, host.vh, format!("su3 {name}: vh"));
        assert_eq!(got_u.data(), host.u.data(), "{name}: u payload bits");
        assert_eq!(got_s.data(), host.s.data(), "{name}: s payload bits");
        assert_eq!(got_vh.data(), host.vh.data(), "{name}: vh payload bits");
        assert_eq!(found.error, host.error, "{name}: truncation error bits");
    }
}

#[test]
fn su3_eigh_composition_matches_host_for_every_policy() {
    let (_, raw) = su3_endomorphism(0x5150_2702);
    let source = raw.add(&raw.adjoint().unwrap(), 1.0, 1.0).unwrap();
    for (name, truncation) in policies() {
        let (d, v) = source.eigh_full().unwrap();
        let bond = d.domain()[0].clone();
        let found = bond
            .find_truncated(&d.diagview().unwrap(), &truncation)
            .unwrap();
        let selection = &found.selection;
        let host = source.eigh_trunc(&truncation).unwrap();

        let got_d = d.restrict_diagonal(selection).unwrap();
        let got_v = v.restrict_leg(v.codomain_rank(), selection).unwrap();

        assert_eq!(*selection.subspace(), host.d.domain()[0], "{name}: bond");
        assert_same_layout!(got_d, host.d, format!("su3 eigh {name}: d"));
        assert_same_layout!(got_v, host.v, format!("su3 eigh {name}: v"));
        assert_eq!(got_d.data(), host.d.data(), "{name}: d payload bits");
        assert_eq!(got_v.data(), host.v.data(), "{name}: v payload bits");
        assert_eq!(found.error, host.error, "{name}: truncation error bits");
    }
}
