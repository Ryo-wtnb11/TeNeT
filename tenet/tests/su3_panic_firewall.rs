//! Public SU(3) operations report unsupported wiring through `Result`, and no
//! public SU(3)-accepting op escapes this firewall (see the census guard at the
//! bottom of the file).

use std::collections::BTreeSet;
use std::panic::{catch_unwind, AssertUnwindSafe};

use tenet::prelude::*;

fn discard<T>(result: Result<T, Error>) -> Result<(), Error> {
    result.map(|_| ())
}

/// Public SU(3)-accepting ops that report their unwired status as a typed,
/// recoverable `Err` (never a panic). Built in one place so both the runtime
/// probe and the census guard read the same op names.
fn unwired_su3_cases<'a>(
    space: &'a Space,
    tensor: &'a Tensor,
    truncation: &'a Truncation,
) -> Vec<(&'static str, Box<dyn Fn() -> Result<(), Error> + 'a>)> {
    vec![
        ("Space::fuse", Box::new(|| discard(space.fuse(space)))),
        (
            "Space::fuse_all",
            Box::new(|| discard(Space::fuse_all(&[space, space]))),
        ),
        ("Tensor::twist", Box::new(|| discard(tensor.twist(&[0])))),
        ("Tensor::flip", Box::new(|| discard(tensor.flip(&[0])))),
        (
            "Tensor::trace_pairs",
            Box::new(|| discard(tensor.trace_pairs(&[(0, 1)]))),
        ),
        ("Tensor::svd_full", Box::new(|| discard(tensor.svd_full()))),
        ("Tensor::qr_full", Box::new(|| discard(tensor.qr_full()))),
        ("Tensor::lq_full", Box::new(|| discard(tensor.lq_full()))),
        (
            "Tensor::left_null",
            Box::new(|| discard(tensor.left_null())),
        ),
        (
            "Tensor::right_null",
            Box::new(|| discard(tensor.right_null())),
        ),
        (
            "Tensor::left_polar",
            Box::new(|| discard(tensor.left_polar())),
        ),
        (
            "Tensor::right_polar",
            Box::new(|| discard(tensor.right_polar())),
        ),
        (
            "Tensor::eigh_full",
            Box::new(|| discard(tensor.eigh_full())),
        ),
        (
            "Tensor::eigh_trunc",
            Box::new(|| discard(tensor.eigh_trunc(truncation))),
        ),
        (
            "Tensor::eigh_vals",
            Box::new(|| discard(tensor.eigh_vals())),
        ),
        ("Tensor::eig_full", Box::new(|| discard(tensor.eig_full()))),
        (
            "Tensor::eig_trunc",
            Box::new(|| discard(tensor.eig_trunc(truncation))),
        ),
        ("Tensor::eig_vals", Box::new(|| discard(tensor.eig_vals()))),
        ("Tensor::exp", Box::new(|| discard(tensor.exp()))),
        ("Tensor::inv", Box::new(|| discard(tensor.inv()))),
        ("Tensor::pinv", Box::new(|| discard(tensor.pinv(1e-12)))),
        // #148: bare Space::sectors panics; try_sectors is the recoverable probe.
        (
            "Space::try_sectors",
            Box::new(|| discard(space.try_sectors())),
        ),
    ]
}

/// Public SU(3)-accepting ops that are fully wired and return `Ok` (or a safe
/// `None`) on a legal SU(3) operand. Kept in sync with the assertions in
/// `wired_su3_result_paths_remain_available`.
const WIRED_SU3_OPS: &[&str] = &[
    "Tensor::tr",
    "Tensor::adjoint",
    "Tensor::norm",
    "Tensor::inner",
    "Tensor::svd_compact",
    "Tensor::svd_trunc",
    "Tensor::svd_vals",
    "Tensor::qr_compact",
    "Tensor::lq_compact",
    "Tensor::permute",
    "Tensor::transpose",
    "Tensor::twist",
    "Tensor::flip",
    "Space::fuse_all",
    "Space::su3_sectors",
    "Space::su3_degeneracy",
    "Space::degeneracy",
];

/// Public SU(3)-accepting ops whose contract is a *documented* panic (bare
/// `Vec` return, no `Result` to carry the error). Pinned by a `#[should_panic]`
/// test so the panic stays a tested contract, not an accident.
const PANIC_SU3_OPS: &[&str] = &["Space::sectors"];

fn su3_space() -> Space {
    Space::su3([((1, 0), 1), ((0, 1), 1)]).unwrap()
}

#[test]
fn unwired_su3_result_apis_return_exact_errors_without_panicking() {
    let runtime = Runtime::builder().build().unwrap();
    let space = su3_space();
    let tensor = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 17).unwrap();
    let truncation = Truncation::Full;

    for (operation, call) in unwired_su3_cases(&space, &tensor, &truncation) {
        let outcome = catch_unwind(AssertUnwindSafe(call));
        let result =
            outcome.unwrap_or_else(|_| panic!("{operation} panicked for a legal SU(3) tensor"));
        assert_eq!(
            result,
            Err(Error::UnsupportedForRule {
                operation,
                rule: "SU(3)",
            }),
            "{operation} returned the wrong recoverable error"
        );
    }
}

#[test]
fn wired_su3_result_paths_remain_available() {
    let runtime = Runtime::builder().build().unwrap();
    let space = su3_space();
    let tensor = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 23).unwrap();

    assert!(tensor.tr().is_ok());
    assert!(tensor.adjoint().is_ok());
    assert!(tensor.norm().is_ok());
    assert!(tensor.svd_compact().is_ok());
    assert!(tensor.svd_trunc(&Truncation::Full).is_ok());
    assert!(tensor.svd_vals().is_ok());
    assert!(tensor.qr_compact().is_ok());
    assert!(tensor.lq_compact().is_ok());
    assert!(tensor.permute(&[0], &[1]).is_ok());
    assert!(tensor.transpose().is_ok());

    assert!(tensor.twist(&[]).is_ok());
    assert!(tensor.flip(&[]).is_ok());
    assert!(Space::fuse_all(&[&space]).is_ok());

    // SU(3)-native read-back accessors and the label-agnostic degeneracy lookup
    // (a non-SU(3) label safely yields None, never a panic).
    assert!(space.su3_sectors().is_ok());
    assert!(space.su3_degeneracy(1, 0).is_ok());
    assert!(space.degeneracy(SectorLabel::U1(0)).is_none());
}

#[test]
#[should_panic(expected = "SU(3) sectors do not fit the `SectorLabel` enum")]
fn bare_sectors_panics_on_su3_as_documented() {
    // sectors() returns a bare Vec, so SU(3) can only surface as a panic; this
    // pins the documented message (see Space::sectors' `# Panics` section).
    let _ = su3_space().sectors();
}

/// Systemic re-occurrence guard (#148): the census of every public
/// SU(3)-accepting Tensor/Space op must exactly match what this firewall
/// exercises. Add a new public SU(3)-accepting op? Add it to `PUBLIC_SU3_OPS`
/// *and* wire a real row above (an unwired case, a wired assertion, or a
/// documented-panic entry). A mismatch either way fails here, so a firewall
/// gap cannot land silently at review time.
const PUBLIC_SU3_OPS: &[&str] = &[
    // unwired -> typed Err
    "Space::fuse",
    "Space::fuse_all",
    "Space::try_sectors",
    "Tensor::twist",
    "Tensor::flip",
    "Tensor::trace_pairs",
    "Tensor::inner",
    "Tensor::svd_full",
    "Tensor::qr_full",
    "Tensor::lq_full",
    "Tensor::left_null",
    "Tensor::right_null",
    "Tensor::left_polar",
    "Tensor::right_polar",
    "Tensor::eigh_full",
    "Tensor::eigh_trunc",
    "Tensor::eigh_vals",
    "Tensor::eig_full",
    "Tensor::eig_trunc",
    "Tensor::eig_vals",
    "Tensor::exp",
    "Tensor::inv",
    "Tensor::pinv",
    // wired -> Ok / safe None
    "Tensor::tr",
    "Tensor::adjoint",
    "Tensor::norm",
    "Tensor::svd_compact",
    "Tensor::svd_trunc",
    "Tensor::svd_vals",
    "Tensor::qr_compact",
    "Tensor::lq_compact",
    "Tensor::permute",
    "Tensor::transpose",
    "Space::su3_sectors",
    "Space::su3_degeneracy",
    "Space::degeneracy",
    // documented panic
    "Space::sectors",
];

#[test]
fn firewall_covers_every_public_su3_op() {
    let space = su3_space();
    let runtime = Runtime::builder().build().unwrap();
    let tensor = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 17).unwrap();
    let truncation = Truncation::Full;

    let mut covered: BTreeSet<&str> = BTreeSet::new();
    for (operation, _) in unwired_su3_cases(&space, &tensor, &truncation) {
        covered.insert(operation);
    }
    covered.extend(WIRED_SU3_OPS.iter().copied());
    covered.extend(PANIC_SU3_OPS.iter().copied());

    let census: BTreeSet<&str> = PUBLIC_SU3_OPS.iter().copied().collect();

    let missing_from_firewall: Vec<_> = census.difference(&covered).collect();
    let missing_from_census: Vec<_> = covered.difference(&census).collect();
    assert!(
        missing_from_firewall.is_empty() && missing_from_census.is_empty(),
        "SU(3) firewall census drift: in census but no firewall row: {missing_from_firewall:?}; \
         exercised by firewall but absent from census: {missing_from_census:?}"
    );
}
