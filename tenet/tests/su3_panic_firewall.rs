//! Public SU(3) operations report unsupported wiring through `Result`.

use std::panic::{catch_unwind, AssertUnwindSafe};

use tenet::prelude::*;

fn discard<T>(result: Result<T, Error>) -> Result<(), Error> {
    result.map(|_| ())
}

#[test]
fn unwired_su3_result_apis_return_exact_errors_without_panicking() {
    let runtime = Runtime::builder().build().unwrap();
    let space = Space::su3([((1, 0), 1), ((0, 1), 1)]).unwrap();
    let tensor = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 17).unwrap();
    let truncation = Truncation::Full;

    let cases: Vec<(&str, Box<dyn Fn() -> Result<(), Error>>)> = vec![
        ("Space::fuse", Box::new(|| discard(space.fuse(&space)))),
        (
            "Space::fuse_all",
            Box::new(|| discard(Space::fuse_all(&[&space, &space]))),
        ),
        ("Tensor::twist", Box::new(|| discard(tensor.twist(&[0])))),
        ("Tensor::flip", Box::new(|| discard(tensor.flip(&[0])))),
        (
            "Tensor::trace_pairs",
            Box::new(|| discard(tensor.trace_pairs(&[(0, 1)]))),
        ),
        ("Tensor::inner", Box::new(|| discard(tensor.inner(&tensor)))),
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
            Box::new(|| discard(tensor.eigh_trunc(&truncation))),
        ),
        (
            "Tensor::eigh_vals",
            Box::new(|| discard(tensor.eigh_vals())),
        ),
        ("Tensor::eig_full", Box::new(|| discard(tensor.eig_full()))),
        (
            "Tensor::eig_trunc",
            Box::new(|| discard(tensor.eig_trunc(&truncation))),
        ),
        ("Tensor::eig_vals", Box::new(|| discard(tensor.eig_vals()))),
        ("Tensor::exp", Box::new(|| discard(tensor.exp()))),
        ("Tensor::inv", Box::new(|| discard(tensor.inv()))),
        ("Tensor::pinv", Box::new(|| discard(tensor.pinv(1e-12)))),
    ];

    for (operation, call) in cases {
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
    let space = Space::su3([((1, 0), 1), ((0, 1), 1)]).unwrap();
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
}
