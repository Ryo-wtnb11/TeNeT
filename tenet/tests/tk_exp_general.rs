//! TensorKit `exp(::AbstractTensorMap)` correspondence for the general
//! (non-Hermitian) arm of `Tensor::exp` / `TensorMap::exp` (issue #577).
//!
//! TensorKit 0.17 `src/tensors/linalg.jl:419-427`:
//!
//! ```text
//! exp(t)  = exp!(copy(t))
//! exp!(t) = (domain(t) == codomain(t) || error(...);
//!            for (c, b) in blocks(t); copy!(b, LinearAlgebra.exp!(b)); end)
//! ```
//!
//! so the only structure in the operation is *blockwise*: the endomorphism
//! check, and then Julia's scaling-and-squaring Padé on each coupled-sector
//! matrix independently. No hermiticity gate, no cross-block coupling.
//!
//! # Oracle provenance
//!
//! The constants below are TensorKit output on Julia 1.11.6 from
//! `~/.julia/packages/TensorKit/6Camk` (0.16.2), for the same fill this file
//! uses:
//!
//! ```julia
//! V = U1Space(0 => 3, 1 => 2)
//! t = zeros(T, V ← V)
//! for (c, b) in blocks(t), j in axes(b, 2), i in axes(b, 1)
//!     re = 0.5 + 0.25 * (i - 1) - 0.75 * (j - 1) + 0.125 * convert(Int, c.charge)
//!     b[i, j] = (T <: Complex) ? complex(re, 0.125 * (i - 1) + 0.375 * (j - 1) - 0.25) : re
//! end
//! b .*= scale
//! norm(t), norm(exp(t))
//! ```
//!
//! The pinned 0.17.0 tree (`~/.julia/packages/TensorKit/jCjQQ`) was not the
//! resolved version. That does not weaken the oracle: `exp!` is
//! character-identical in the two trees and contains no arithmetic of its own,
//! so these are Julia stdlib v1.11 numbers in either.
//!
//! # Fixture certification
//!
//! `norm` is a pre-existing method that predates #577, and `norm(t)` is
//! basis-free — `sqrt(Σ_c dim(c) ||block_c||_F²)`, with `dim(c) = 1` for every
//! U(1) sector — so it depends on the multiset of block entries and nothing
//! else. Pinning `norm` of the *input* against Julia therefore proves TeNeT's
//! fixture is the tensor Julia exponentiated, before any of #577's code runs.
//! The entrywise oracle for the same fixture lives one layer down, in
//! `tenet-matrixalgebra`'s `exp_of_a_multisector_u1_endomorphism_matches_the_tensorkit_oracle`.

use std::sync::Arc;

use tenet::core::{U1FusionRule, U1Irrep};
use tenet::prelude::{BlockKey, Complex64, Runtime, Space, Tensor};
use tenet::typed::{BlockFusionTrees, GradedSpace, TensorMap};

/// Relative agreement with the TensorKit oracle. The two engines evaluate the
/// same approximant at different Padé degrees for the small blocks (Julia drops
/// to degree 3/5/7/9 below `||A||_1 = 2.1`; TeNeT always uses [13/13]), so this
/// is approximant agreement, not bitwise agreement — and `1e-12` is orders of
/// magnitude tighter than any coefficient, scaling or dispatch mistake.
const RTOL: f64 = 1e-12;

fn runtime() -> Runtime {
    Runtime::builder().dense_threads(1).build().unwrap()
}

fn real_fill(charge: i32, indices: &[usize], scale: f64) -> f64 {
    scale * (0.5 + 0.25 * indices[0] as f64 - 0.75 * indices[1] as f64 + 0.125 * charge as f64)
}

fn imaginary_fill(indices: &[usize], scale: f64) -> f64 {
    scale * (0.125 * indices[0] as f64 + 0.375 * indices[1] as f64 - 0.25)
}

fn erased_charge(key: &BlockKey) -> i32 {
    let pair = key.as_fusion_tree_pair().expect("fusion-tree block");
    U1Irrep::from_sector_id(pair.codomain_tree().coupled())
        .expect("built-in U(1) codec decodes its own ids")
        .charge()
}

fn erased_space() -> Space {
    Space::u1([(0, 3), (1, 2)])
}

fn typed_space() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
        false,
    )
    .unwrap()
}

fn erased_real(runtime: &Runtime, scale: f64) -> Tensor {
    let space = erased_space();
    Tensor::from_block_fn(runtime, [&space], [&space], |key, indices| {
        real_fill(erased_charge(key), indices, scale)
    })
    .unwrap()
}

fn erased_complex(runtime: &Runtime, scale: f64) -> Tensor {
    let space = erased_space();
    Tensor::from_block_fn(runtime, [&space], [&space], |key, indices| {
        Complex64::new(
            real_fill(erased_charge(key), indices, scale),
            imaginary_fill(indices, scale),
        )
    })
    .unwrap()
}

fn typed_real(runtime: &Runtime, scale: f64) -> TensorMap<U1FusionRule, f64> {
    let leg = typed_space();
    TensorMap::from_block_fn(
        runtime,
        [&leg],
        [&leg],
        |sectors: &BlockFusionTrees<U1Irrep>, indices: &[usize]| {
            real_fill(sectors.coupled().charge(), indices, scale)
        },
    )
    .unwrap()
}

fn typed_complex(runtime: &Runtime, scale: f64) -> TensorMap<U1FusionRule, Complex64> {
    let leg = typed_space();
    TensorMap::from_block_fn(
        runtime,
        [&leg],
        [&leg],
        |sectors: &BlockFusionTrees<U1Irrep>, indices: &[usize]| {
            Complex64::new(
                real_fill(sectors.coupled().charge(), indices, scale),
                imaginary_fill(indices, scale),
            )
        },
    )
    .unwrap()
}

fn assert_close(actual: f64, expected: f64, what: &str) {
    let relative = (actual - expected).abs() / expected.abs();
    assert!(
        relative <= RTOL,
        "{what}: {actual:.17e} differs from the TensorKit oracle {expected:.17e} by {relative:e}"
    );
}

#[test]
fn general_exp_matches_the_tensorkit_oracle_on_both_facades() {
    let runtime = runtime();

    // (scale, norm(t), norm(exp(t))) from the Julia session quoted above.
    for (scale, input_norm, exponential_norm) in [
        (1.0, 2.2220486043288972, 3.1532168621506798),
        (4.0, 8.8881944173155887, 15.692503963067274),
    ] {
        let erased = erased_real(&runtime, scale);
        let typed = typed_real(&runtime, scale);
        assert_close(
            erased.norm().unwrap(),
            input_norm,
            &format!("f64 scale {scale} input fixture"),
        );
        assert_close(
            typed.norm().unwrap(),
            input_norm,
            &format!("f64 scale {scale} typed input fixture"),
        );

        let erased_exp = erased.exp().unwrap();
        let typed_exp = typed.exp().unwrap();
        assert_close(
            erased_exp.norm().unwrap(),
            exponential_norm,
            &format!("f64 scale {scale} exp"),
        );
        assert_eq!(
            typed_exp.data(),
            erased_exp.data(),
            "f64 scale {scale}: the two facades published different bytes"
        );
    }

    // c64, where the blocks are non-Hermitian in both parts.
    let erased = erased_complex(&runtime, 1.0);
    let typed = typed_complex(&runtime, 1.0);
    assert_close(
        erased.norm().unwrap(),
        2.5678298230217673,
        "c64 input fixture",
    );
    assert_close(
        typed.norm().unwrap(),
        2.5678298230217673,
        "c64 typed input fixture",
    );

    let erased_exp = erased.exp().unwrap();
    let typed_exp = typed.exp().unwrap();
    assert_close(erased_exp.norm().unwrap(), 3.1806015158373824, "c64 exp");
    assert_eq!(
        typed_exp.data(),
        erased_exp.data_c64(),
        "c64: the two facades published different bytes"
    );
}
