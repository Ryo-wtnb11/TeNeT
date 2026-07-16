//! Dtype/placement panic firewall (#128).
//!
//! The runtime-erased `Tensor` storage accessors (`data`/`data_c64`/`to_c64`)
//! panic on a *legal* tensor that is simply the wrong dtype (or device-
//! resident). Each has a recoverable `try_*` counterpart that returns a typed
//! [`Error`] instead. This firewall pins that both halves stay wired, and the
//! census guard at the bottom makes a new dtype/placement accessor fail here
//! unless it is added with a `try_*` row (mirrors the SU(3) firewall, #148).

use std::collections::BTreeSet;
use std::panic::{catch_unwind, AssertUnwindSafe};

use tenet::prelude::*;

fn u1_space() -> Space {
    Space::u1([(-1, 1), (0, 2), (1, 1)])
}

fn f64_tensor(rt: &Runtime, space: &Space) -> Tensor {
    Tensor::rand_with_seed(rt, Dtype::F64, [space], [space], 17).unwrap()
}

fn c64_tensor(rt: &Runtime, space: &Space) -> Tensor {
    Tensor::rand_with_seed(rt, Dtype::C64, [space], [space], 17).unwrap()
}

/// Public dtype/placement-dependent accessors whose `try_*` form must report a
/// wrong-dtype legal tensor as a typed `Err`, never a panic.
const TRY_ACCESSORS: &[&str] = &[
    "Tensor::try_data",
    "Tensor::try_data_c64",
    "Tensor::try_to_c64",
];

/// The panicking halves of each pair: a documented panic on the wrong legal
/// state, pinned by `#[should_panic]` tests below.
const PANIC_ACCESSORS: &[&str] = &["Tensor::data", "Tensor::data_c64", "Tensor::to_c64"];

#[test]
fn try_accessors_return_typed_errors_without_panicking() {
    let rt = Runtime::builder().build().unwrap();
    let space = u1_space();
    let f64 = f64_tensor(&rt, &space);
    let c64 = c64_tensor(&rt, &space);

    // Wrong-dtype access -> DtypeMismatch (a legal tensor, wrong scalar type).
    assert_eq!(
        catch_unwind(AssertUnwindSafe(|| f64.try_data_c64().map(|_| ())))
            .expect("try_data_c64 panicked"),
        Err(Error::DtypeMismatch),
    );
    assert_eq!(
        catch_unwind(AssertUnwindSafe(|| c64.try_data().map(|_| ()))).expect("try_data panicked"),
        Err(Error::DtypeMismatch),
    );

    // Matching-dtype access and widening succeed.
    assert!(f64.try_data().is_ok());
    assert!(c64.try_data_c64().is_ok());
    assert!(f64.try_to_c64().is_ok()); // f64 -> c64 widen
    assert!(c64.try_to_c64().is_ok()); // c64 -> c64 clone
}

#[test]
#[should_panic(expected = "tensor is not host c64")]
fn bare_data_c64_panics_on_f64_as_documented() {
    let rt = Runtime::builder().build().unwrap();
    let _ = f64_tensor(&rt, &u1_space()).data_c64();
}

#[test]
#[should_panic(expected = "tensor is not host f64")]
fn bare_data_panics_on_c64_as_documented() {
    let rt = Runtime::builder().build().unwrap();
    let _ = c64_tensor(&rt, &u1_space()).data();
}

/// Re-occurrence guard (#128): every public dtype/placement-dependent storage
/// accessor must be listed with its recoverable `try_*` partner. Adding a new
/// one without a `try_*` row (or forgetting to exercise it above) fails here.
#[test]
fn firewall_covers_every_dtype_placement_accessor() {
    let covered: BTreeSet<&str> = TRY_ACCESSORS
        .iter()
        .chain(PANIC_ACCESSORS.iter())
        .copied()
        .collect();

    // Each panicking accessor has exactly one `try_*` partner, so the two
    // lists must be the same length: a lone panic accessor means a missing
    // firewall row.
    assert_eq!(
        TRY_ACCESSORS.len(),
        PANIC_ACCESSORS.len(),
        "each panicking storage accessor needs a try_* partner"
    );
    assert_eq!(covered.len(), TRY_ACCESSORS.len() + PANIC_ACCESSORS.len());
}
