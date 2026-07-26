//! TensorKit `norm(t, p)` correspondence for `Tensor::norm_p` / `TensorMap::norm_p`
//! (issue #597, item 2).
//!
//! TensorKit 0.17 `src/tensors/linalg.jl:257-275`:
//!
//! ```text
//! p == Inf     -> maximum entry magnitude over blocks(t)
//! finite p > 0 -> (sum_c dim(c) * norm(block_c, p)^p)^(1/p)
//! otherwise    -> ArgumentError
//! ```
//!
//! `norm(block, p)` is Julia's *entrywise* p-norm (matrices included), so only
//! the multiset of a coupled sector's entries and its quantum dimension enter
//! the result — never the block's shape or the engine's fusion-tree ordering.
//! That is what makes the oracle below portable between TensorKit's storage and
//! TeNeT's.
//!
//! # Oracle provenance
//!
//! The constants are TensorKit 0.17.0 output (`~/.julia/packages/TensorKit/jCjQQ`,
//! Julia 1.11.6), from a tensor built block-by-block with exactly the fill this
//! file uses:
//!
//! ```julia
//! realfill(i, j) = 1.0 + 0.5 * (i - 1) - 0.25 * (j - 1)
//! cplxfill(i, j) = realfill(i, j) + im * (0.5 + 0.125 * (i - 1) + 0.375 * (j - 1))
//! t = zeros(T, V <- W); for (c, b) in blocks(t), j in axes(b, 2), i in axes(b, 1)
//!     b[i, j] = fill(i, j)
//! end
//! norm(t, p)
//! ```
//!
//! with `V = U1Space(-1 => 2, 0 => 3, 1 => 4)`, `W = U1Space(-1 => 5, 0 => 1, 1 => 2)`
//! and the SU(2) pair `V = SU2Space(0 => 2, 1//2 => 3, 1 => 4)`,
//! `W = SU2Space(0 => 5, 1//2 => 1, 1 => 2)`.
//!
//! The degeneracies are deliberately all distinct, so no two coupled sectors
//! carry the same entry multiset: a `dim(c)` weight attached to the wrong
//! sector changes the answer.

use std::sync::Arc;

use tenet::core::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use tenet::prelude::{Complex64, Runtime, Space, Tensor};
use tenet::typed::{GradedSpace, TensorMap};

/// Relative agreement with the TensorKit oracle. The two engines sum the same
/// terms in different orders, so exact equality is not the claim; `1e-13` is
/// far tighter than any weighting or dispatch mistake could survive.
const RTOL: f64 = 1e-13;

fn runtime() -> Runtime {
    Runtime::builder().dense_threads(1).build().unwrap()
}

/// The real fill, in the engine's 0-based degeneracy coordinates.
///
/// `1.0 + 0.5 * i - 0.25 * j` hits exactly zero at `(0, 4)`, which the
/// 5-fold domain sector reaches — so the fixtures also cover a `0^p` term.
fn real_fill(indices: &[usize]) -> f64 {
    1.0 + 0.5 * indices[0] as f64 - 0.25 * indices[1] as f64
}

fn complex_fill(indices: &[usize]) -> Complex64 {
    Complex64::new(
        real_fill(indices),
        0.5 + 0.125 * indices[0] as f64 + 0.375 * indices[1] as f64,
    )
}

fn assert_close(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= RTOL * expected.abs(),
        "{what}: got {actual:.17e}, TensorKit says {expected:.17e}"
    );
}

// TensorKit 0.17.0 oracle values, indexed as [p = 1, 2, 3, Inf].
const U1_F64: [f64; 4] = [25.0, 6.2048368229954285, 4.080404211522764, 2.5];
const U1_C64: [f64; 4] = [
    36.08814448804253,
    8.145167278822456,
    5.0601391475683535,
    2.6487025125521364,
];
const SU2_F64: [f64; 4] = [55.5, 9.656603957913982, 5.588779616741119, 2.5];
const SU2_C64: [f64; 4] = [
    70.83774683619109,
    11.637090486887175,
    6.51405041399105,
    2.6487025125521364,
];

// ---------------------------------------------------------------------------
// Erased facade
// ---------------------------------------------------------------------------

fn erased_u1() -> (Space, Space) {
    (
        Space::u1([(-1, 2), (0, 3), (1, 4)]),
        Space::u1([(-1, 5), (0, 1), (1, 2)]),
    )
}

fn erased_su2() -> (Space, Space) {
    (
        Space::su2([(0, 2), (1, 3), (2, 4)]).unwrap(),
        Space::su2([(0, 5), (1, 1), (2, 2)]).unwrap(),
    )
}

fn erased_cases(rt: &Runtime) -> Vec<(&'static str, Tensor, [f64; 4])> {
    let (u1_v, u1_w) = erased_u1();
    let (su2_v, su2_w) = erased_su2();
    vec![
        (
            "u1 f64",
            Tensor::from_block_fn(rt, [&u1_v], [&u1_w], |_, indices| real_fill(indices)).unwrap(),
            U1_F64,
        ),
        (
            "u1 c64",
            Tensor::from_block_fn(rt, [&u1_v], [&u1_w], |_, indices| complex_fill(indices))
                .unwrap(),
            U1_C64,
        ),
        (
            "su2 f64",
            Tensor::from_block_fn(rt, [&su2_v], [&su2_w], |_, indices| real_fill(indices)).unwrap(),
            SU2_F64,
        ),
        (
            "su2 c64",
            Tensor::from_block_fn(rt, [&su2_v], [&su2_w], |_, indices| complex_fill(indices))
                .unwrap(),
            SU2_C64,
        ),
    ]
}

// ---------------------------------------------------------------------------
// Typed facade
// ---------------------------------------------------------------------------

fn typed_u1() -> (GradedSpace<U1FusionRule>, GradedSpace<U1FusionRule>) {
    let rule = Arc::new(U1FusionRule);
    let pairs = |entries: [(i32, usize); 3]| {
        GradedSpace::try_new(
            Arc::clone(&rule),
            entries.map(|(charge, deg)| (U1Irrep::new(charge), deg)),
            false,
        )
        .unwrap()
    };
    (
        pairs([(-1, 2), (0, 3), (1, 4)]),
        pairs([(-1, 5), (0, 1), (1, 2)]),
    )
}

fn typed_su2() -> (GradedSpace<SU2FusionRule>, GradedSpace<SU2FusionRule>) {
    let rule = Arc::new(SU2FusionRule);
    let pairs = |entries: [(usize, usize); 3]| {
        GradedSpace::try_new(
            Arc::clone(&rule),
            entries.map(|(twice_spin, deg)| (SU2Irrep::from_twice_spin(twice_spin), deg)),
            false,
        )
        .unwrap()
    };
    (
        pairs([(0, 2), (1, 3), (2, 4)]),
        pairs([(0, 5), (1, 1), (2, 2)]),
    )
}

#[test]
fn the_erased_fixtures_are_the_tensors_tensorkit_measured() {
    // `norm_p` does not exist yet; what this pins is the *fixture*, which is
    // the part an oracle table cannot check itself. TensorKit's `norm(t, 2)`
    // and `norm(t, Inf)` are the two rows the facade can already compute, so
    // agreement there is what certifies that the tensor built here carries the
    // same per-sector entry multisets as the one Julia measured — and hence
    // that the p = 1 and p = 3 rows are usable as oracles for `norm_p`.
    let rt = runtime();
    for (name, tensor, oracle) in erased_cases(&rt) {
        assert_close(
            tensor.norm().unwrap(),
            oracle[1],
            &format!("{name} norm() vs TensorKit norm(t, 2)"),
        );
        assert_close(
            tensor.norm_inf().unwrap(),
            oracle[3],
            &format!("{name} norm_inf() vs TensorKit norm(t, Inf)"),
        );
    }
}

#[test]
fn the_typed_fixtures_are_the_tensors_tensorkit_measured() {
    // Same certification for the typed facade: it builds its own tensors from
    // its own leg type, so its fixtures need their own anchor.
    let rt = runtime();
    let (u1_v, u1_w) = typed_u1();
    let (su2_v, su2_w) = typed_su2();

    let u1_f64: TensorMap<U1FusionRule, f64> =
        TensorMap::from_block_fn(&rt, [&u1_v], [&u1_w], |_, indices| real_fill(indices)).unwrap();
    let u1_c64: TensorMap<U1FusionRule, Complex64> =
        TensorMap::from_block_fn(&rt, [&u1_v], [&u1_w], |_, indices| complex_fill(indices))
            .unwrap();
    let su2_f64: TensorMap<SU2FusionRule, f64> =
        TensorMap::from_block_fn(&rt, [&su2_v], [&su2_w], |_, indices| real_fill(indices)).unwrap();
    let su2_c64: TensorMap<SU2FusionRule, Complex64> =
        TensorMap::from_block_fn(&rt, [&su2_v], [&su2_w], |_, indices| complex_fill(indices))
            .unwrap();

    assert_close(u1_f64.norm().unwrap(), U1_F64[1], "typed u1 f64 norm()");
    assert_close(
        u1_f64.norm_inf().unwrap(),
        U1_F64[3],
        "typed u1 f64 norm_inf()",
    );
    assert_close(u1_c64.norm().unwrap(), U1_C64[1], "typed u1 c64 norm()");
    assert_close(
        u1_c64.norm_inf().unwrap(),
        U1_C64[3],
        "typed u1 c64 norm_inf()",
    );
    assert_close(su2_f64.norm().unwrap(), SU2_F64[1], "typed su2 f64 norm()");
    assert_close(
        su2_f64.norm_inf().unwrap(),
        SU2_F64[3],
        "typed su2 f64 norm_inf()",
    );
    assert_close(su2_c64.norm().unwrap(), SU2_C64[1], "typed su2 c64 norm()");
    assert_close(
        su2_c64.norm_inf().unwrap(),
        SU2_C64[3],
        "typed su2 c64 norm_inf()",
    );
}
