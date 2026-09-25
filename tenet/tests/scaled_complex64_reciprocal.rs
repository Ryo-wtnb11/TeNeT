//! The `Complex64` compact reciprocal (TeNeT#1463): `inv`/`pinv` of a
//! multiplicity-free compact (diagonal) `Complex64` tensor apply
//! `ScalarOps::recip_value` elementwise (`tenet/src/typed.rs`). Before this
//! fix that was the naive `1.0 / z`, i.e. `(c - d*i) / (c*c + d*d)`: squaring
//! both components overflows `f64` for `|z| >=~ 2^512` and flushes the
//! denominator to zero for `|z| <=~ 2^-511`, although `1/z` is finite and
//! representable across that whole range (#1459 hit the same failure one
//! `f32` squaring away, at `Complex32`, and #1462 fixed it there by dividing
//! in `Complex64`; there is no wider type for `Complex64` itself).
//!
//! TeNeT's semantics reference for `inv` is TensorKit, which delegates a
//! diagonal entry's reciprocal to the host language's own scalar `inv`
//! (Julia's `LinearAlgebra`/`Base`). `tenet/src/typed.rs`'s
//! `julia_complex64_reciprocal` is accordingly a literal, line-cited port of
//! Julia 1.11.6's `Base.inv(w::ComplexF64)` (`base/complex.jl:479`) and its
//! `robust_cinv` (Baudin-Smith) fallback (`base/complex.jl:508`) — not an
//! independently designed algorithm, and not tuned for correct rounding: the
//! fast path is exact where it applies, and the scaled fallback is exact in
//! range without claiming last-bit accuracy beyond what Julia itself
//! guarantees. `julia_complex32_reciprocal_wide` likewise ports
//! `inv(z::Complex{Float32})` (`base/complex.jl:473`).
//!
//! Oracle: this crate's own reciprocal is compared **bitwise** against
//! Julia 1.11.6 actually running `inv` on the same inputs
//! (`benchmarks/complex64_reciprocal_oracle.jl`, output committed at
//! `benchmarks/complex64_reciprocal_oracle.out`), not against an
//! independently-rounded value — since the Rust code is a literal port of
//! the same operation sequence (`muladd`/`mul_add`, `copysign`/`flipsign`),
//! any mismatch is a porting bug, not an acceptable rounding difference.

use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use tenet::core::{U1FusionRule, U1Irrep};
use tenet::prelude::{GradedSpace, Runtime, SectorSpectrum, TensorMap};

fn runtime() -> Runtime {
    Runtime::builder().build().expect("runtime builds")
}

fn leg(dim: usize) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), dim)]).unwrap()
}

fn bits(z: Complex64) -> (u64, u64) {
    (z.re.to_bits(), z.im.to_bits())
}

fn bits32(z: Complex32) -> (u32, u32) {
    (z.re.to_bits(), z.im.to_bits())
}

/// Builds a rank-`(1,1)` compact diagonal tensor holding exactly `value` and
/// returns the single stored entry of `t.inv()`.
fn compact_inv(value: Complex64) -> Complex64 {
    let rt = runtime();
    let space = leg(1);
    let diagonal: TensorMap<_, Complex64> = TensorMap::diagonal(
        &rt,
        &space,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![value],
        }],
    )
    .unwrap();
    diagonal.inv().unwrap().diagview().unwrap()[0].values[0]
}

/// As [`compact_inv`], but also returns `t.pinv(0.0)`'s entry (`rcond = 0`
/// never discards a nonzero entry). Only valid for a finite `value`: `pinv`'s
/// own preflight requires every stored magnitude to be finite (unrelated to
/// this issue), so it is never asked to invert an infinite entry.
fn compact_inv_and_pinv(value: Complex64) -> (Complex64, Complex64) {
    let rt = runtime();
    let space = leg(1);
    let diagonal: TensorMap<_, Complex64> = TensorMap::diagonal(
        &rt,
        &space,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![value],
        }],
    )
    .unwrap();
    let inv = diagonal.inv().unwrap().diagview().unwrap()[0].values[0];
    let pinv = diagonal.pinv(0.0).unwrap().diagview().unwrap()[0].values[0];
    (inv, pinv)
}

/// As [`compact_inv`], but for a `Complex32` payload (`julia_complex32_reciprocal_wide`).
fn compact_inv32(value: Complex32) -> Complex32 {
    let rt = runtime();
    let space = leg(1);
    let diagonal: TensorMap<_, Complex32> = TensorMap::diagonal(
        &rt,
        &space,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![value],
        }],
    )
    .unwrap();
    diagonal.inv().unwrap().diagview().unwrap()[0].values[0]
}

/// One oracle-checked `(z, 1/z)` pair, bits taken verbatim from
/// `benchmarks/complex64_reciprocal_oracle.out` (Julia 1.11.6's own `inv`).
/// `z_re`/`z_im`/`exp_re`/`exp_im` are `f64` bit patterns
/// (`f64::from_bits`) rather than float literals so every input — including
/// a subnormal, an infinite component, or the exact top/bottom of the
/// exponent range — is reproduced bit-for-bit. `check_pinv` is `false` for
/// the infinite-component cases, which `pinv`'s finite-magnitude preflight
/// rejects for a reason unrelated to this issue (see [`compact_inv_and_pinv`]).
struct OracleCase {
    label: &'static str,
    z_re: u64,
    z_im: u64,
    exp_re: u64,
    exp_im: u64,
    check_pinv: bool,
}

/// Generated by `benchmarks/complex64_reciprocal_oracle.jl` (Julia 1.11.6,
/// Base only): `2^{+-}600`, the naive algorithm's old `2^{+-}511`/`2^{+-}512`
/// squaring-overflow boundary, the largest/smallest normal `2^{+-}1022`,
/// true subnormal input down to the smallest subnormal, purely real and
/// purely imaginary entries of each magnitude, twelve `inf`-component
/// combinations, and a random sweep across those regimes (`random_*`
/// labels). Regenerate by rerunning that script and re-embedding its output
/// here; do not hand-edit a row.
const ORACLE_CASES: &[OracleCase] = &[
    OracleCase {
        label: "inf_re_pos_im_zero",
        z_re: 0x7ff0000000000000,
        z_im: 0x0000000000000000,
        exp_re: 0x0000000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: false,
    },
    OracleCase {
        label: "inf_re_neg_im_zero",
        z_re: 0xfff0000000000000,
        z_im: 0x0000000000000000,
        exp_re: 0x8000000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: false,
    },
    OracleCase {
        label: "re_zero_inf_im_pos",
        z_re: 0x0000000000000000,
        z_im: 0x7ff0000000000000,
        exp_re: 0x0000000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: false,
    },
    OracleCase {
        label: "re_zero_inf_im_neg",
        z_re: 0x0000000000000000,
        z_im: 0xfff0000000000000,
        exp_re: 0x0000000000000000,
        exp_im: 0x0000000000000000,
        check_pinv: false,
    },
    OracleCase {
        label: "inf_re_pos_inf_im_pos",
        z_re: 0x7ff0000000000000,
        z_im: 0x7ff0000000000000,
        exp_re: 0x0000000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: false,
    },
    OracleCase {
        label: "inf_re_pos_inf_im_neg",
        z_re: 0x7ff0000000000000,
        z_im: 0xfff0000000000000,
        exp_re: 0x0000000000000000,
        exp_im: 0x0000000000000000,
        check_pinv: false,
    },
    OracleCase {
        label: "inf_re_neg_inf_im_pos",
        z_re: 0xfff0000000000000,
        z_im: 0x7ff0000000000000,
        exp_re: 0x8000000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: false,
    },
    OracleCase {
        label: "inf_re_neg_inf_im_neg",
        z_re: 0xfff0000000000000,
        z_im: 0xfff0000000000000,
        exp_re: 0x8000000000000000,
        exp_im: 0x0000000000000000,
        check_pinv: false,
    },
    OracleCase {
        label: "inf_re_pos_finite_im_pos",
        z_re: 0x7ff0000000000000,
        z_im: 0x4008000000000000,
        exp_re: 0x0000000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: false,
    },
    OracleCase {
        label: "inf_re_neg_finite_im_neg",
        z_re: 0xfff0000000000000,
        z_im: 0xc008000000000000,
        exp_re: 0x8000000000000000,
        exp_im: 0x0000000000000000,
        check_pinv: false,
    },
    OracleCase {
        label: "finite_re_pos_inf_im_pos",
        z_re: 0x4008000000000000,
        z_im: 0x7ff0000000000000,
        exp_re: 0x0000000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: false,
    },
    OracleCase {
        label: "finite_re_neg_inf_im_neg",
        z_re: 0xc008000000000000,
        z_im: 0xfff0000000000000,
        exp_re: 0x8000000000000000,
        exp_im: 0x0000000000000000,
        check_pinv: false,
    },
    OracleCase {
        label: "real_2p600",
        z_re: 0x6570000000000000,
        z_im: 0x0000000000000000,
        exp_re: 0x1a70000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "neg_real_2p600",
        z_re: 0xe570000000000000,
        z_im: 0x0000000000000000,
        exp_re: 0x9a70000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "imag_2p600",
        z_re: 0x0000000000000000,
        z_im: 0x6570000000000000,
        exp_re: 0x0000000000000000,
        exp_im: 0x9a70000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "neg_imag_2p600",
        z_re: 0x0000000000000000,
        z_im: 0xe570000000000000,
        exp_re: 0x0000000000000000,
        exp_im: 0x1a70000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "mixed_2p600",
        z_re: 0x6570000000000000,
        z_im: 0x6570000000000000,
        exp_re: 0x1a60000000000000,
        exp_im: 0x9a60000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "mixed_neg_2p600",
        z_re: 0x6570000000000000,
        z_im: 0xe570000000000000,
        exp_re: 0x1a60000000000000,
        exp_im: 0x1a60000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "real_2p-600",
        z_re: 0x1a70000000000000,
        z_im: 0x0000000000000000,
        exp_re: 0x6570000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "neg_real_2p-600",
        z_re: 0x9a70000000000000,
        z_im: 0x0000000000000000,
        exp_re: 0xe570000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "imag_2p-600",
        z_re: 0x0000000000000000,
        z_im: 0x1a70000000000000,
        exp_re: 0x0000000000000000,
        exp_im: 0xe570000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "neg_imag_2p-600",
        z_re: 0x0000000000000000,
        z_im: 0x9a70000000000000,
        exp_re: 0x0000000000000000,
        exp_im: 0x6570000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "mixed_2p-600",
        z_re: 0x1a70000000000000,
        z_im: 0x1a70000000000000,
        exp_re: 0x6560000000000000,
        exp_im: 0xe560000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "mixed_neg_2p-600",
        z_re: 0x1a70000000000000,
        z_im: 0x9a70000000000000,
        exp_re: 0x6560000000000000,
        exp_im: 0x6560000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "real_2p511",
        z_re: 0x5fe0000000000000,
        z_im: 0x0000000000000000,
        exp_re: 0x2000000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "neg_real_2p511",
        z_re: 0xdfe0000000000000,
        z_im: 0x0000000000000000,
        exp_re: 0xa000000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "imag_2p511",
        z_re: 0x0000000000000000,
        z_im: 0x5fe0000000000000,
        exp_re: 0x0000000000000000,
        exp_im: 0xa000000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "neg_imag_2p511",
        z_re: 0x0000000000000000,
        z_im: 0xdfe0000000000000,
        exp_re: 0x0000000000000000,
        exp_im: 0x2000000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "mixed_2p511",
        z_re: 0x5fe0000000000000,
        z_im: 0x5fe0000000000000,
        exp_re: 0x1ff0000000000000,
        exp_im: 0x9ff0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "mixed_neg_2p511",
        z_re: 0x5fe0000000000000,
        z_im: 0xdfe0000000000000,
        exp_re: 0x1ff0000000000000,
        exp_im: 0x1ff0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "real_2p512",
        z_re: 0x5ff0000000000000,
        z_im: 0x0000000000000000,
        exp_re: 0x1ff0000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "neg_real_2p512",
        z_re: 0xdff0000000000000,
        z_im: 0x0000000000000000,
        exp_re: 0x9ff0000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "imag_2p512",
        z_re: 0x0000000000000000,
        z_im: 0x5ff0000000000000,
        exp_re: 0x0000000000000000,
        exp_im: 0x9ff0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "neg_imag_2p512",
        z_re: 0x0000000000000000,
        z_im: 0xdff0000000000000,
        exp_re: 0x0000000000000000,
        exp_im: 0x1ff0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "mixed_2p512",
        z_re: 0x5ff0000000000000,
        z_im: 0x5ff0000000000000,
        exp_re: 0x1fe0000000000000,
        exp_im: 0x9fe0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "mixed_neg_2p512",
        z_re: 0x5ff0000000000000,
        z_im: 0xdff0000000000000,
        exp_re: 0x1fe0000000000000,
        exp_im: 0x1fe0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "real_2p-511",
        z_re: 0x2000000000000000,
        z_im: 0x0000000000000000,
        exp_re: 0x5fe0000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "neg_real_2p-511",
        z_re: 0xa000000000000000,
        z_im: 0x0000000000000000,
        exp_re: 0xdfe0000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "imag_2p-511",
        z_re: 0x0000000000000000,
        z_im: 0x2000000000000000,
        exp_re: 0x0000000000000000,
        exp_im: 0xdfe0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "neg_imag_2p-511",
        z_re: 0x0000000000000000,
        z_im: 0xa000000000000000,
        exp_re: 0x0000000000000000,
        exp_im: 0x5fe0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "mixed_2p-511",
        z_re: 0x2000000000000000,
        z_im: 0x2000000000000000,
        exp_re: 0x5fd0000000000000,
        exp_im: 0xdfd0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "mixed_neg_2p-511",
        z_re: 0x2000000000000000,
        z_im: 0xa000000000000000,
        exp_re: 0x5fd0000000000000,
        exp_im: 0x5fd0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "real_2p-512",
        z_re: 0x1ff0000000000000,
        z_im: 0x0000000000000000,
        exp_re: 0x5ff0000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "neg_real_2p-512",
        z_re: 0x9ff0000000000000,
        z_im: 0x0000000000000000,
        exp_re: 0xdff0000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "imag_2p-512",
        z_re: 0x0000000000000000,
        z_im: 0x1ff0000000000000,
        exp_re: 0x0000000000000000,
        exp_im: 0xdff0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "neg_imag_2p-512",
        z_re: 0x0000000000000000,
        z_im: 0x9ff0000000000000,
        exp_re: 0x0000000000000000,
        exp_im: 0x5ff0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "mixed_2p-512",
        z_re: 0x1ff0000000000000,
        z_im: 0x1ff0000000000000,
        exp_re: 0x5fe0000000000000,
        exp_im: 0xdfe0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "mixed_neg_2p-512",
        z_re: 0x1ff0000000000000,
        z_im: 0x9ff0000000000000,
        exp_re: 0x5fe0000000000000,
        exp_im: 0x5fe0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "real_2p1022",
        z_re: 0x7fd0000000000000,
        z_im: 0x0000000000000000,
        exp_re: 0x0010000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "neg_real_2p1022",
        z_re: 0xffd0000000000000,
        z_im: 0x0000000000000000,
        exp_re: 0x8010000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "imag_2p1022",
        z_re: 0x0000000000000000,
        z_im: 0x7fd0000000000000,
        exp_re: 0x0000000000000000,
        exp_im: 0x8010000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "neg_imag_2p1022",
        z_re: 0x0000000000000000,
        z_im: 0xffd0000000000000,
        exp_re: 0x0000000000000000,
        exp_im: 0x0010000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "mixed_2p1022",
        z_re: 0x7fd0000000000000,
        z_im: 0x7fd0000000000000,
        exp_re: 0x0008000000000000,
        exp_im: 0x8008000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "mixed_neg_2p1022",
        z_re: 0x7fd0000000000000,
        z_im: 0xffd0000000000000,
        exp_re: 0x0008000000000000,
        exp_im: 0x0008000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "real_2p-1022",
        z_re: 0x0010000000000000,
        z_im: 0x0000000000000000,
        exp_re: 0x7fd0000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "neg_real_2p-1022",
        z_re: 0x8010000000000000,
        z_im: 0x0000000000000000,
        exp_re: 0xffd0000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "imag_2p-1022",
        z_re: 0x0000000000000000,
        z_im: 0x0010000000000000,
        exp_re: 0x0000000000000000,
        exp_im: 0xffd0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "neg_imag_2p-1022",
        z_re: 0x0000000000000000,
        z_im: 0x8010000000000000,
        exp_re: 0x0000000000000000,
        exp_im: 0x7fd0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "mixed_2p-1022",
        z_re: 0x0010000000000000,
        z_im: 0x0010000000000000,
        exp_re: 0x7fc0000000000000,
        exp_im: 0xffc0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "mixed_neg_2p-1022",
        z_re: 0x0010000000000000,
        z_im: 0x8010000000000000,
        exp_re: 0x7fc0000000000000,
        exp_im: 0x7fc0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "real_2p-1030_subnormal",
        z_re: 0x0000100000000000,
        z_im: 0x0000000000000000,
        exp_re: 0x7ff0000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "neg_real_2p-1030_subnormal",
        z_re: 0x8000100000000000,
        z_im: 0x0000000000000000,
        exp_re: 0xfff0000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "imag_2p-1030_subnormal",
        z_re: 0x0000000000000000,
        z_im: 0x0000100000000000,
        exp_re: 0x0000000000000000,
        exp_im: 0xfff0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "neg_imag_2p-1030_subnormal",
        z_re: 0x0000000000000000,
        z_im: 0x8000100000000000,
        exp_re: 0x0000000000000000,
        exp_im: 0x7ff0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "mixed_2p-1030_subnormal",
        z_re: 0x0000100000000000,
        z_im: 0x0000100000000000,
        exp_re: 0x7ff0000000000000,
        exp_im: 0xfff0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "mixed_neg_2p-1030_subnormal",
        z_re: 0x0000100000000000,
        z_im: 0x8000100000000000,
        exp_re: 0x7ff0000000000000,
        exp_im: 0x7ff0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "real_min_subnormal",
        z_re: 0x0000000000000001,
        z_im: 0x0000000000000000,
        exp_re: 0x7ff0000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "neg_real_min_subnormal",
        z_re: 0x8000000000000001,
        z_im: 0x0000000000000000,
        exp_re: 0xfff0000000000000,
        exp_im: 0x8000000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "imag_min_subnormal",
        z_re: 0x0000000000000000,
        z_im: 0x0000000000000001,
        exp_re: 0x0000000000000000,
        exp_im: 0xfff0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "neg_imag_min_subnormal",
        z_re: 0x0000000000000000,
        z_im: 0x8000000000000001,
        exp_re: 0x0000000000000000,
        exp_im: 0x7ff0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "mixed_min_subnormal",
        z_re: 0x0000000000000001,
        z_im: 0x0000000000000001,
        exp_re: 0x7ff0000000000000,
        exp_im: 0xfff0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "mixed_neg_min_subnormal",
        z_re: 0x0000000000000001,
        z_im: 0x8000000000000001,
        exp_re: 0x7ff0000000000000,
        exp_im: 0x7ff0000000000000,
        check_pinv: true,
    },
    OracleCase {
        label: "random_0_e-600",
        z_re: 0x9a74cfe900c1a500,
        z_im: 0x9a75f550b8be5d74,
        exp_re: 0xe55748937890e618,
        exp_im: 0x655890d2603315d2,
        check_pinv: true,
    },
    OracleCase {
        label: "random_1_e0",
        z_re: 0xbfcb94ada8d8b6e0,
        z_im: 0xbff08b1b9c37253a,
        exp_re: 0xbfc8b9924520dc4a,
        exp_im: 0x3feda928076bfa12,
        check_pinv: true,
    },
    OracleCase {
        label: "random_2_e600",
        z_re: 0x657c7fa94e666c6e,
        z_im: 0x6544c9ff7ba6d240,
        exp_re: 0x1a61d14e4227a0d0,
        exp_im: 0x9a29feb6121cbed2,
        check_pinv: true,
    },
    OracleCase {
        label: "random_3_e0",
        z_re: 0x3fc2bf85a17ded20,
        z_im: 0xbfe06c0e79448bf4,
        exp_re: 0x3fe074cc8c88b823,
        exp_im: 0x3ffcd440fa11eccc,
        check_pinv: true,
    },
    OracleCase {
        label: "random_4_e300",
        z_re: 0x52a01bf2feff8c6c,
        z_im: 0x5287d2c1929fd8e0,
        exp_re: 0x2d3bf60c6fe285fb,
        exp_im: 0xad24ace3069aa0c5,
        check_pinv: true,
    },
    OracleCase {
        label: "random_5_e0",
        z_re: 0xbffc3a4d108260f2,
        z_im: 0x3ff5a5c5b0fc374a,
        exp_re: 0xbfd6d7a45fbfd9d9,
        exp_im: 0xbfd1847e5abf641c,
        check_pinv: true,
    },
    OracleCase {
        label: "random_6_e-511",
        z_re: 0x200d5c421858e204,
        z_im: 0x2006d56f3f5eeab2,
        exp_re: 0x5fc5bb87573e5636,
        exp_im: 0xdfc0e6c949b8e224,
        check_pinv: true,
    },
    OracleCase {
        label: "random_7_e300",
        z_re: 0xd2810502014008b0,
        z_im: 0xd2b3ddc653ace686,
        exp_re: 0xacf5d43316d6eeea,
        exp_im: 0x2d297ae64365bfc3,
        check_pinv: true,
    },
    OracleCase {
        label: "random_8_e600",
        z_re: 0xe55606afbfb70768,
        z_im: 0xe574ebdb893c703e,
        exp_re: 0x9a481882acfdde83,
        exp_im: 0x1a66e31b3264a953,
        check_pinv: true,
    },
    OracleCase {
        label: "random_9_e511",
        z_re: 0xdfc981ba11426db0,
        z_im: 0x5feef280c2a756ca,
        exp_re: 0x9fca292e91262ab1,
        exp_im: 0x9fefbdad03df1927,
        check_pinv: true,
    },
    OracleCase {
        label: "random_10_e300",
        z_re: 0x52b1c2e360931d74,
        z_im: 0xd2bf46f5ef83e43e,
        exp_re: 0x2d0c1dd63de180d9,
        exp_im: 0x2d18c19bd81e33ac,
        check_pinv: true,
    },
    OracleCase {
        label: "random_11_e-511",
        z_re: 0x1fed0f254120eaf0,
        z_im: 0x1ff0c4746b0cbf5c,
        exp_re: 0x5fde398c37f9ffa7,
        exp_im: 0xdfe170a2b35e8b5a,
        check_pinv: true,
    },
    OracleCase {
        label: "random_12_e0",
        z_re: 0x3ffb49eeeb5ae988,
        z_im: 0x3fea9ba1e943e740,
        exp_re: 0x3fde5185afaf9830,
        exp_im: 0xbfcd8fdec933c518,
        check_pinv: true,
    },
    OracleCase {
        label: "random_13_e0",
        z_re: 0xbffd5f2889d72b10,
        z_im: 0xbfdfba02e2a09900,
        exp_re: 0xbfe03f3823e9476e,
        exp_im: 0x3fc18cafe835b71f,
        check_pinv: true,
    },
    OracleCase {
        label: "random_14_e511",
        z_re: 0xdfd28fe66fe6bccc,
        z_im: 0xdfe0ff9b1aefc19e,
        exp_re: 0x9fe95672434c0cec,
        exp_im: 0x1ff7340938ab90a3,
        check_pinv: true,
    },
    OracleCase {
        label: "random_15_e-300",
        z_re: 0x2d23f03b405fa458,
        z_im: 0xad3f8a713e322fda,
        exp_re: 0x5282a8bbde1c50c1,
        exp_im: 0x529d845ec23d510d,
        check_pinv: true,
    },
    OracleCase {
        label: "random_16_e10",
        z_re: 0xc07354f7b5ea1380,
        z_im: 0xc0751584d374f818,
        exp_re: 0xbf583150d6df5199,
        exp_im: 0x3f5a62a61e11ac44,
        check_pinv: true,
    },
    OracleCase {
        label: "random_17_e10",
        z_re: 0x4073c665ae974b80,
        z_im: 0xc09b7d4e3904e454,
        exp_re: 0x3f19f510dd8a4139,
        exp_im: 0x3f420ab278c65543,
        check_pinv: true,
    },
    OracleCase {
        label: "random_18_e0",
        z_re: 0xbfe4c5267265c76c,
        z_im: 0xbfe70431b0e7be50,
        exp_re: 0xbfe620d4b3c25583,
        exp_im: 0x3fe88579dfe67d60,
        check_pinv: true,
    },
    OracleCase {
        label: "random_19_e-300",
        z_re: 0x2d30cb8128873cee,
        z_im: 0xad0505aa1c5793d0,
        exp_re: 0x52adc1c536518348,
        exp_im: 0x52829f8f247b657f,
        check_pinv: true,
    },
    OracleCase {
        label: "random_20_e300",
        z_re: 0xd2b44f8b1890dcc4,
        z_im: 0x52a91630985e2bbc,
        exp_re: 0xad223f9c1ec22141,
        exp_im: 0xad168a1d8bf8839a,
        check_pinv: true,
    },
    OracleCase {
        label: "random_21_e-600",
        z_re: 0x9a7bf20fc8f219bc,
        z_im: 0x1a59edd0092c4258,
        exp_re: 0xe56162c996d0509f,
        exp_im: 0xe540219c7865cfa6,
        check_pinv: true,
    },
    OracleCase {
        label: "random_22_e-600",
        z_re: 0x9a74c1b6375593c0,
        z_im: 0x1a66e4e2bd2e9f20,
        exp_re: 0xe562ea0a46b602b9,
        exp_im: 0xe554dca4e7f89ab9,
        check_pinv: true,
    },
    OracleCase {
        label: "random_23_e300",
        z_re: 0x52ab993847b2762c,
        z_im: 0x52bc1f9afaa9a1ec,
        exp_re: 0x2d0ccc4a37d6f5b0,
        exp_im: 0xad1d58842271d796,
        check_pinv: true,
    },
    OracleCase {
        label: "random_24_e10",
        z_re: 0x408a00f8294a51fc,
        z_im: 0xc070aa570cd9fe98,
        exp_re: 0x3f51db1efc0714ed,
        exp_im: 0x3f36e31c10841b0d,
        check_pinv: true,
    },
    OracleCase {
        label: "random_25_e0",
        z_re: 0xbfc051c84f558ca0,
        z_im: 0xbfff687b59b97728,
        exp_re: 0xbfa0de812cbfc0e5,
        exp_im: 0x3fe03ba810b1a516,
        check_pinv: true,
    },
    OracleCase {
        label: "random_26_e0",
        z_re: 0xbff0123c8364d6e6,
        z_im: 0x3ff3db1f0ce9c3c8,
        exp_re: 0xbfd9383ec04f4c47,
        exp_im: 0xbfdf28a9b94c6837,
        check_pinv: true,
    },
    OracleCase {
        label: "random_27_e10",
        z_re: 0xc08213a901e2b644,
        z_im: 0xc09485872822f1e4,
        exp_re: 0xbf3268117a7b87af,
        exp_im: 0x3f44e55a109f0bd6,
        check_pinv: true,
    },
    OracleCase {
        label: "random_28_e-511",
        z_re: 0xa00456974153cb2a,
        z_im: 0xa005069789b3138e,
        exp_re: 0xdfc8566fc8801bde,
        exp_im: 0x5fc9290c1f34e827,
        check_pinv: true,
    },
    OracleCase {
        label: "random_29_e-511",
        z_re: 0xa00edc55e1bf4ff6,
        z_im: 0xa0021486827ec8ae,
        exp_re: 0xdfc8b3da03838351,
        exp_im: 0x5fbcf1d6da25a8c7,
        check_pinv: true,
    },
    OracleCase {
        label: "random_30_e511",
        z_re: 0x5fd8b1c735c96f58,
        z_im: 0xdfe33825d5430a96,
        exp_re: 0x1fe83a9d55e16963,
        exp_im: 0x1ff2db678e384eef,
        check_pinv: true,
    },
    OracleCase {
        label: "random_31_e-511",
        z_re: 0x200a44509566a184,
        z_im: 0xa00bbaadf7637ea2,
        exp_re: 0x5fc26fed2b7bd94e,
        exp_im: 0x5fc376b347bc3ce6,
        check_pinv: true,
    },
    OracleCase {
        label: "random_32_e10",
        z_re: 0x407f781e13b9c4a8,
        z_im: 0x40849787a41ddfac,
        exp_re: 0x3f47fd9b74229a3a,
        exp_im: 0xbf4f6589226e673b,
        check_pinv: true,
    },
    OracleCase {
        label: "random_33_e10",
        z_re: 0xc08e3dd91c91618c,
        z_im: 0xc0926f6959a27e18,
        exp_re: 0xbf3b3c7d34abcf9f,
        exp_im: 0x3f409a6983e15646,
        check_pinv: true,
    },
    OracleCase {
        label: "random_34_e300",
        z_re: 0xd2a98e853409012c,
        z_im: 0xd290910ae59cb0e8,
        exp_re: 0xad32211f328e8122,
        exp_im: 0x2d178101cc80b963,
        check_pinv: true,
    },
    OracleCase {
        label: "random_35_e300",
        z_re: 0xd28ff8a0d8aa1ef0,
        z_im: 0x52b47153aae5d706,
        exp_re: 0xad02dd32145996cc,
        exp_im: 0xad281faa18ee7182,
        check_pinv: true,
    },
    OracleCase {
        label: "random_36_e-600",
        z_re: 0x9a63837bf45fa1a0,
        z_im: 0x9a6620972a9166b4,
        exp_re: 0xe566f50c953d5bb8,
        exp_im: 0x656a083bf61e8887,
        check_pinv: true,
    },
    OracleCase {
        label: "random_37_e-600",
        z_re: 0x1a50cc6d3d0cd928,
        z_im: 0x1a7a0a0d622d33b2,
        exp_re: 0x6538b9f8c45209d4,
        exp_im: 0xe5632a0409d1468a,
        check_pinv: true,
    },
    OracleCase {
        label: "random_38_e-511",
        z_re: 0xa0092d6a0d9edfb4,
        z_im: 0x200a5edbb6bad690,
        exp_re: 0xdfc365127372c34d,
        exp_im: 0xdfc4505d1564e007,
        check_pinv: true,
    },
    OracleCase {
        label: "random_39_e10",
        z_re: 0x407d64fd1ea24430,
        z_im: 0xc05b8459a2ddd700,
        exp_re: 0x3f608385a5222733,
        exp_im: 0x3f3eeb002d0555dd,
        check_pinv: true,
    },
    OracleCase {
        label: "random_40_e300",
        z_re: 0xd2b6ef59fed7ad36,
        z_im: 0xd29c11f84d0ae980,
        exp_re: 0xad2469b45b59940e,
        exp_im: 0x2d08fbc2911ab873,
        check_pinv: true,
    },
    OracleCase {
        label: "random_41_e0",
        z_re: 0x3fdab7db2f18f5f0,
        z_im: 0x3fe7695b4fa62c2c,
        exp_re: 0x3fe2d3f135323be9,
        exp_im: 0xbff07f6d4c250ef5,
        check_pinv: true,
    },
    OracleCase {
        label: "random_42_e-600",
        z_re: 0x9a7235c7e978dc70,
        z_im: 0x1a68f48459e3fd94,
        exp_re: 0xe563221b2764c314,
        exp_im: 0xe55a3860ac76b3cf,
        check_pinv: true,
    },
    OracleCase {
        label: "random_43_e-600",
        z_re: 0x1a61626e3c285444,
        z_im: 0x1a6e14b9ebe0cb80,
        exp_re: 0x655d7ed21cf66c79,
        exp_im: 0xe56984b771f97fa5,
        check_pinv: true,
    },
    OracleCase {
        label: "random_44_e-511",
        z_re: 0xa000d7621b35fa04,
        z_im: 0xa00d0f8786814440,
        exp_re: 0xdfbe92b304e16382,
        exp_im: 0x5fca60b354d147d1,
        check_pinv: true,
    },
    OracleCase {
        label: "random_45_e-600",
        z_re: 0x9a7aa2c4c1447966,
        z_im: 0x1a78f77de9566642,
        exp_re: 0xe55476e2e593e868,
        exp_im: 0xe5532e9ba263510c,
        check_pinv: true,
    },
    OracleCase {
        label: "random_46_e-600",
        z_re: 0x9a7120e7e0652758,
        z_im: 0x9a76bc32c73fa850,
        exp_re: 0xe555a583741328da,
        exp_im: 0x655cbb64c0f4b793,
        check_pinv: true,
    },
    OracleCase {
        label: "random_47_e10",
        z_re: 0xc09f91984fe2ab34,
        z_im: 0x4068752648f1b8a0,
        exp_re: 0xbf40116117c9f3d4,
        exp_im: 0xbf08e59d24794fe1,
        check_pinv: true,
    },
    OracleCase {
        label: "random_48_e-300",
        z_re: 0xad0421f184c1cb60,
        z_im: 0x2d1a2098916e3c40,
        exp_re: 0xd2aa4bf5a8b31771,
        exp_im: 0xd2c1103e49dd4c68,
        check_pinv: true,
    },
    OracleCase {
        label: "random_49_e511",
        z_re: 0x5fb7e2bb97b0d290,
        z_im: 0xdfe155c918eb7880,
        exp_re: 0x1fd3c30c7b04f7c1,
        exp_im: 0x1ffcaf3d4a06741e,
        check_pinv: true,
    },
    OracleCase {
        label: "random_50_e300",
        z_re: 0x52b609d8cba7d96c,
        z_im: 0x52b99d2b4549eac4,
        exp_re: 0x2d13c3ead3ca0e4b,
        exp_im: 0xad16f8d40eda39f9,
        check_pinv: true,
    },
    OracleCase {
        label: "random_51_e511",
        z_re: 0x5fe33067d7ba564e,
        z_im: 0x5fe6f0a4056dfbb2,
        exp_re: 0x1fe5f7d0798a1cd8,
        exp_im: 0x9fea431c09bd7143,
        check_pinv: true,
    },
    OracleCase {
        label: "random_52_e600",
        z_re: 0xe574365e31a7cc0e,
        z_im: 0x653ac29f1f7d0e20,
        exp_re: 0x9a6928a401f79805,
        exp_im: 0x9a30a78dc341a597,
        check_pinv: true,
    },
    OracleCase {
        label: "random_53_e600",
        z_re: 0xe5793b0fecf04826,
        z_im: 0x65711ef0f61433e2,
        exp_re: 0x9a5bca1c4cd5fa59,
        exp_im: 0x9a52db7060417092,
        check_pinv: true,
    },
    OracleCase {
        label: "random_54_e-511",
        z_re: 0x200bbdde16fac6a6,
        z_im: 0x200545904fefd94e,
        exp_re: 0x5fc73ebb8d6b5934,
        exp_im: 0xdfc1d2e5c322b9ba,
        check_pinv: true,
    },
    OracleCase {
        label: "random_55_e0",
        z_re: 0x3fdeee3dc56012a8,
        z_im: 0x3ff0f58a4774e290,
        exp_re: 0x3fd6cadd68bd2150,
        exp_im: 0xbfe8fe7761873be0,
        check_pinv: true,
    },
    OracleCase {
        label: "random_56_e600",
        z_re: 0xe576087191ebfe18,
        z_im: 0xe570dc2b12a86340,
        exp_re: 0x9a5d4fdaaa37a04c,
        exp_im: 0x1a566e1552274208,
        check_pinv: true,
    },
    OracleCase {
        label: "random_57_e-300",
        z_re: 0xad31e8353f823c54,
        z_im: 0xad0bf0b6f187a630,
        exp_re: 0xd2ab8b5774add1c8,
        exp_im: 0x52857d15b8a294a1,
        check_pinv: true,
    },
    OracleCase {
        label: "random_58_e-300",
        z_re: 0x2d2efcfae7023a00,
        z_im: 0x2d312fb54a27ce96,
        exp_re: 0x529da1b934cef0c0,
        exp_im: 0xd2a06f1dc28ef114,
        check_pinv: true,
    },
    OracleCase {
        label: "random_59_e0",
        z_re: 0xbfe8c28834b5b1a8,
        z_im: 0xbffa680e609c1ba6,
        exp_re: 0xbfcdceff5658b4b7,
        exp_im: 0x3fdfca790d14b303,
        check_pinv: true,
    },
];

/// `inv` (and, where valid, `pinv`) of a compact diagonal entry match
/// Julia's `inv(::ComplexF64)` **bitwise** at every case above.
#[test]
fn compact_reciprocal_matches_julia_bitwise() {
    for case in ORACLE_CASES {
        let z = Complex64::new(f64::from_bits(case.z_re), f64::from_bits(case.z_im));
        let expected = Complex64::new(f64::from_bits(case.exp_re), f64::from_bits(case.exp_im));
        if case.check_pinv {
            let (inv, pinv) = compact_inv_and_pinv(z);
            assert_eq!(
                bits(inv),
                bits(expected),
                "inv({}) = {inv:?}, expected {expected:?}",
                case.label
            );
            assert_eq!(
                bits(pinv),
                bits(expected),
                "pinv({}) = {pinv:?}, expected {expected:?}",
                case.label
            );
        } else {
            let inv = compact_inv(z);
            assert_eq!(
                bits(inv),
                bits(expected),
                "inv({}) = {inv:?}, expected {expected:?}",
                case.label
            );
        }
    }
}

/// A `Complex32` oracle-checked `(z, 1/z)` pair, bits taken verbatim from
/// `benchmarks/complex64_reciprocal_oracle.out`'s `c32_*`-labeled rows
/// (Julia 1.11.6's `inv(::ComplexF32)`). Large, small-normal, subnormal, and
/// infinite `f32` magnitudes, purely real/imaginary and mixed: this path
/// (`julia_complex32_reciprocal_wide`) only ever reaches the widened
/// multiply-based branch, never the `ComplexF64` fast-path/`robust_cinv`
/// split [`ORACLE_CASES`] exercises, but is still checked bitwise here.
struct OracleCase32 {
    label: &'static str,
    z_re: u32,
    z_im: u32,
    exp_re: u32,
    exp_im: u32,
}

const ORACLE_CASES_32: &[OracleCase32] = &[
    OracleCase32 {
        label: "c32_large_real",
        z_re: 0x7f000000,
        z_im: 0x00000000,
        exp_re: 0x00400000,
        exp_im: 0x80000000,
    },
    OracleCase32 {
        label: "c32_large_imag",
        z_re: 0x00000000,
        z_im: 0x7f000000,
        exp_re: 0x00000000,
        exp_im: 0x80400000,
    },
    OracleCase32 {
        label: "c32_large_mixed",
        z_re: 0x7f000000,
        z_im: 0xff000000,
        exp_re: 0x00200000,
        exp_im: 0x00200000,
    },
    OracleCase32 {
        label: "c32_small_normal_real",
        z_re: 0x00800000,
        z_im: 0x00000000,
        exp_re: 0x7e800000,
        exp_im: 0x80000000,
    },
    OracleCase32 {
        label: "c32_small_normal_imag",
        z_re: 0x00000000,
        z_im: 0x00800000,
        exp_re: 0x00000000,
        exp_im: 0xfe800000,
    },
    OracleCase32 {
        label: "c32_small_normal_mixed",
        z_re: 0x00800000,
        z_im: 0x80800000,
        exp_re: 0x7e000000,
        exp_im: 0x7e000000,
    },
    OracleCase32 {
        label: "c32_subnormal_real",
        z_re: 0x00000010,
        z_im: 0x00000000,
        exp_re: 0x7f800000,
        exp_im: 0x80000000,
    },
    OracleCase32 {
        label: "c32_subnormal_imag",
        z_re: 0x00000000,
        z_im: 0x00000010,
        exp_re: 0x00000000,
        exp_im: 0xff800000,
    },
    OracleCase32 {
        label: "c32_subnormal_min_mixed",
        z_re: 0x00000001,
        z_im: 0x80000001,
        exp_re: 0x7f800000,
        exp_im: 0x7f800000,
    },
    OracleCase32 {
        label: "c32_inf_re_pos_im_zero",
        z_re: 0x7f800000,
        z_im: 0x00000000,
        exp_re: 0x00000000,
        exp_im: 0x80000000,
    },
    OracleCase32 {
        label: "c32_inf_re_neg_im_zero",
        z_re: 0xff800000,
        z_im: 0x00000000,
        exp_re: 0x80000000,
        exp_im: 0x80000000,
    },
    OracleCase32 {
        label: "c32_re_zero_inf_im_pos",
        z_re: 0x00000000,
        z_im: 0x7f800000,
        exp_re: 0x00000000,
        exp_im: 0x80000000,
    },
    OracleCase32 {
        label: "c32_re_zero_inf_im_neg",
        z_re: 0x00000000,
        z_im: 0xff800000,
        exp_re: 0x00000000,
        exp_im: 0x00000000,
    },
    OracleCase32 {
        label: "c32_inf_re_pos_inf_im_pos",
        z_re: 0x7f800000,
        z_im: 0x7f800000,
        exp_re: 0x00000000,
        exp_im: 0x80000000,
    },
    OracleCase32 {
        label: "c32_inf_re_pos_finite_im",
        z_re: 0x7f800000,
        z_im: 0x40400000,
        exp_re: 0x00000000,
        exp_im: 0x80000000,
    },
    OracleCase32 {
        label: "c32_finite_re_inf_im_pos",
        z_re: 0x40400000,
        z_im: 0x7f800000,
        exp_re: 0x00000000,
        exp_im: 0x80000000,
    },
];

/// `inv` of a compact `Complex32` diagonal entry matches Julia's
/// `inv(::ComplexF32)` **bitwise** at every case above.
#[test]
fn compact_reciprocal_32_matches_julia_bitwise() {
    for case in ORACLE_CASES_32 {
        let z = Complex32::new(f32::from_bits(case.z_re), f32::from_bits(case.z_im));
        let expected = Complex32::new(f32::from_bits(case.exp_re), f32::from_bits(case.exp_im));
        let inv = compact_inv32(z);
        assert_eq!(
            bits32(inv),
            bits32(expected),
            "inv({}) = {inv:?}, expected {expected:?}",
            case.label
        );
    }
}

/// `inv`/`pinv` of a singular (exactly zero) compact entry is still the
/// caller-mistake `InvalidArgument` the compact arm has always reported
/// (`tenet/src/typed.rs::inv_multiplicity_free`), unaffected by which
/// reciprocal algorithm the nonzero branch uses. The literal Julia port is
/// never reached for a zero entry because of this preflight, so it is not
/// itself required to special-case zero (Julia's own `inv(0.0+0.0im)` is
/// `NaN + NaN*im`, not an error).
#[test]
fn zero_entry_is_still_reported_as_a_singular_diagonal() {
    let rt = runtime();
    let space = leg(1);
    let diagonal: TensorMap<_, Complex64> = TensorMap::diagonal(
        &rt,
        &space,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![Complex64::new(0.0, 0.0)],
        }],
    )
    .unwrap();
    assert!(diagonal.inv().is_err());
}
