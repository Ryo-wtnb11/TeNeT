//! Default-runtime ergonomics: `default!(rt)` / `set_default_runtime` lets the
//! argument-free constructors (`zeros`/`rand`/`id`) skip the runtime argument,
//! while explicit `Tensor::zeros(&rt, …)` and `rt.zeros(…)` keep working for
//! juggling several runtimes.

use tenet::prelude::*;

fn v() -> Space {
    Space::u1([(-1, 1), (0, 2), (1, 1)])
}

#[test]
fn default_free_functions_match_explicit_construction() {
    let rt = Runtime::builder().build().unwrap();
    default!(rt); // == set_default_runtime(&rt)

    let s = v();
    // zeros: deterministic, so all three forms are byte-identical.
    let free = zeros(Dtype::F64, [&s], [&s]).unwrap();
    let method = rt.zeros(Dtype::F64, [&s], [&s]).unwrap();
    let explicit = Tensor::zeros(&rt, Dtype::F64, [&s], [&s]).unwrap();
    assert_eq!(free.data(), explicit.data());
    assert_eq!(method.data(), explicit.data());

    // Seeded rand: same seed + same runtime ⇒ identical across all forms.
    let free = rand_with_seed(Dtype::F64, [&s], [&s], 7).unwrap();
    let method = rt.rand_with_seed(Dtype::F64, [&s], [&s], 7).unwrap();
    let explicit = Tensor::rand_with_seed(&rt, Dtype::F64, [&s], [&s], 7).unwrap();
    assert_eq!(free.data(), explicit.data());
    assert_eq!(method.data(), explicit.data());

    // id and rand free functions at least build against the default.
    assert!(id(Dtype::F64, [&s]).is_ok());
    assert!(rand(Dtype::F64, [&s], [&s]).is_ok());
}

#[test]
fn free_functions_error_without_a_default_runtime() {
    // Thread-local default persists across tests on a reused harness thread, so
    // clear it explicitly before asserting the unset behavior.
    clear_default_runtime();
    assert!(default_runtime().is_err());
    assert!(matches!(
        zeros(Dtype::F64, [&v()], [&v()]),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn explicit_runtimes_are_independent_of_the_default() {
    // The default drives the argument-free path; an explicit runtime drives its
    // own. Tensors from different runtimes must not silently mix.
    let rt = Runtime::builder().build().unwrap();
    let rt2 = Runtime::builder().build().unwrap();
    set_default_runtime(&rt);

    let a = zeros(Dtype::F64, [&v()], [&v()]).unwrap(); // on the default (rt)
    let b = rt2.zeros(Dtype::F64, [&v()], [&v()]).unwrap(); // on rt2
    assert!(matches!(a.compose(&b), Err(Error::RuntimeMismatch)));
}
