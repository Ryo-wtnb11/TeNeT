// The TensorKit predicates and projections as the public chains TeNeT spells
// them with (#1557). Included with `include!` so the integration tests and the
// crate's own unit tests read one spelling; `TensorMap` must be in scope at the
// call site, and the includer implements `ChainCoefficient` for its payloads. Each macro evaluates to a `bool` (or the projected tensor) and
// panics on an operation error.

/// The exact `±1` and `±1/2` coefficients the chains need. Integration tests
/// implement it with `common/predicate_chain_coefficients.rs`; the crate's own
/// unit tests implement it over the crate-private scalar trait.
#[allow(dead_code)]
trait ChainCoefficient: Copy {
    fn real(value: f64) -> Self;
}

/// `‖t - t†‖ <= tol · max(‖t‖, 1)`; a non-endomorphism is `false`.
#[allow(unused_macros)]
macro_rules! is_hermitian {
    ($tensor:expr, $tol:expr) => {{
        let t = &$tensor;
        t.codomain() == t.domain()
            && t.axpby(
                ChainCoefficient::real(1.0),
                &t.adjoint().unwrap(),
                ChainCoefficient::real(-1.0),
            )
            .unwrap()
            .norm(2.0)
            .unwrap()
                <= ($tol) * t.norm(2.0).unwrap().max(1.0)
    }};
}

/// `‖t + t†‖ <= tol · max(‖t‖, 1)`; a non-endomorphism is `false`.
#[allow(unused_macros)]
macro_rules! is_antihermitian {
    ($tensor:expr, $tol:expr) => {{
        let t = &$tensor;
        t.codomain() == t.domain()
            && t.axpby(
                ChainCoefficient::real(1.0),
                &t.adjoint().unwrap(),
                ChainCoefficient::real(1.0),
            )
            .unwrap()
            .norm(2.0)
            .unwrap()
                <= ($tol) * t.norm(2.0).unwrap().max(1.0)
    }};
}

/// `‖t†t - id‖ <= tol · max(‖t†t‖, 1)` on the domain.
#[allow(unused_macros)]
macro_rules! is_isometric {
    ($tensor:expr, $tol:expr) => {{
        let t = &$tensor;
        let gram = t.adjoint().unwrap().compose(t).unwrap();
        let identity = TensorMap::id(t.runtime(), &t.domain()).unwrap();
        gram.axpby(
            ChainCoefficient::real(1.0),
            &identity,
            ChainCoefficient::real(-1.0),
        )
        .unwrap()
        .norm(2.0)
        .unwrap()
            <= ($tol) * gram.norm(2.0).unwrap().max(1.0)
    }};
}

/// Isometric in both directions.
#[allow(unused_macros)]
macro_rules! is_unitary {
    ($tensor:expr, $tol:expr) => {{
        let t = &$tensor;
        is_isometric!(t, $tol) && is_isometric!(t.adjoint().unwrap(), $tol)
    }};
}

/// Hermitian, and every `eigh_vals` eigenvalue strictly above
/// `tol · max(‖t‖, 1)`.
#[allow(unused_macros)]
macro_rules! is_posdef {
    ($tensor:expr, $tol:expr) => {{
        let t = &$tensor;
        is_hermitian!(t, $tol) && {
            let threshold = ($tol) * t.norm(2.0).unwrap().max(1.0);
            t.eigh_vals()
                .unwrap()
                .iter()
                .flat_map(|spectrum| spectrum.values.iter())
                .all(|&value| value > threshold)
        }
    }};
}

/// [`is_posdef!`] for a compact spectrum factor: its stored values are its
/// Hermitian eigenvalues, so `diagview` replaces `eigh_vals`; `$re` reads the
/// real part of one stored value as `f64`.
#[allow(unused_macros)]
macro_rules! is_posdef_compact {
    ($tensor:expr, $tol:expr, $re:expr) => {{
        let t = &$tensor;
        is_hermitian!(t, $tol) && {
            let threshold = ($tol) * t.norm(2.0).unwrap().max(1.0);
            let re = $re;
            t.diagview()
                .unwrap()
                .iter()
                .flat_map(|spectrum| spectrum.values.iter())
                .all(|&value| re(value) > threshold)
        }
    }};
}

/// `(t + t†)/2`.
#[allow(unused_macros)]
macro_rules! project_hermitian {
    ($tensor:expr) => {{
        let t = &$tensor;
        t.adjoint().and_then(|adjoint| {
            t.axpby(
                ChainCoefficient::real(0.5),
                &adjoint,
                ChainCoefficient::real(0.5),
            )
        })
    }};
}

/// `(t - t†)/2`.
#[allow(unused_macros)]
macro_rules! project_antihermitian {
    ($tensor:expr) => {{
        let t = &$tensor;
        t.adjoint().and_then(|adjoint| {
            t.axpby(
                ChainCoefficient::real(0.5),
                &adjoint,
                ChainCoefficient::real(-0.5),
            )
        })
    }};
}
