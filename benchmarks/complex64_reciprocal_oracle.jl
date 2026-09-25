# Julia reference oracle for `Base.inv(::ComplexF64)`, TeNeT's TensorKit/Julia
# semantics reference for the compact `inv`/`pinv` reciprocal of a diagonal
# `Complex64` entry (issue #1463).
#
# `tenet/src/typed.rs`'s `julia_complex64_reciprocal` is a literal,
# line-cited port of `Base.inv(w::ComplexF64)` (`base/complex.jl:479`) and
# its `robust_cinv` kernel (`base/complex.jl:508`) — the Baudin-Smith
# fallback used once the fast path's safe-square range
# (`sqrt(floatmin/2) <= max(|re|,|im|) <= sqrt(floatmax/2)`) does not apply.
# Because the Rust port matches Julia's operation order and constants
# exactly (`muladd` -> `mul_add`, `copysign`/`flipsign` -> `f64::copysign`),
# `tenet/tests/scaled_complex64_reciprocal.rs` checks it against this
# script's output *bitwise*, not within a tolerance.
#
# The Float32 path (`inv(z::Complex{Float32})`, `base/complex.jl:473`,
# ported as `julia_complex32_reciprocal_wide`) gets its own, separate
# ComplexF32 case list below: it only ever reaches the widened
# multiply-based branch, so it never needs the ComplexF64 fast-path/
# robust_cinv split the main case list above probes, but is still checked
# bitwise here rather than only against the decimal-tolerance oracle in
# `tenet/tests/single_precision_advanced_linalg.rs`.
#
# Run (Julia 1.11.6, Base only, no packages):
#   julia benchmarks/complex64_reciprocal_oracle.jl > benchmarks/complex64_reciprocal_oracle.out
#
# The committed .out fixture and its SHA-256 manifest live next to this
# script; `tenet/tests/scaled_complex64_reciprocal.rs` embeds the same
# (label, input bits, output bits) rows as Rust constants rather than
# reading the fixture file at test time, so re-running this script after an
# intentional change to the case list means re-embedding its output too.

bits(x::Float64) = reinterpret(UInt64, x)
frombits(u::UInt64) = reinterpret(Float64, u)
hex16(u::UInt64) = string(u; base=16, pad=16)

# (label, re bits, im bits) — the same 2^{+-}600, 2^{+-}511/512 (the naive
# algorithm's old squaring-overflow boundary), 2^{+-}1022 (largest/smallest
# normal), true-subnormal, purely-real/imaginary, and random-sweep cases
# `tenet/tests/scaled_complex64_reciprocal.rs` uses, plus infinite-component
# cases the coordinator asked to add (Julia's `inv` returns signed zeros for
# those, not NaN).
const CASES = [
    ("inf_re_pos_im_zero", 0x7ff0000000000000, 0x0000000000000000),
    ("inf_re_neg_im_zero", 0xfff0000000000000, 0x0000000000000000),
    ("re_zero_inf_im_pos", 0x0000000000000000, 0x7ff0000000000000),
    ("re_zero_inf_im_neg", 0x0000000000000000, 0xfff0000000000000),
    ("inf_re_pos_inf_im_pos", 0x7ff0000000000000, 0x7ff0000000000000),
    ("inf_re_pos_inf_im_neg", 0x7ff0000000000000, 0xfff0000000000000),
    ("inf_re_neg_inf_im_pos", 0xfff0000000000000, 0x7ff0000000000000),
    ("inf_re_neg_inf_im_neg", 0xfff0000000000000, 0xfff0000000000000),
    ("inf_re_pos_finite_im_pos", 0x7ff0000000000000, 0x4008000000000000),
    ("inf_re_neg_finite_im_neg", 0xfff0000000000000, 0xc008000000000000),
    ("finite_re_pos_inf_im_pos", 0x4008000000000000, 0x7ff0000000000000),
    ("finite_re_neg_inf_im_neg", 0xc008000000000000, 0xfff0000000000000),
    ("real_2p600", 0x6570000000000000, 0x0000000000000000),
    ("neg_real_2p600", 0xe570000000000000, 0x0000000000000000),
    ("imag_2p600", 0x0000000000000000, 0x6570000000000000),
    ("neg_imag_2p600", 0x0000000000000000, 0xe570000000000000),
    ("mixed_2p600", 0x6570000000000000, 0x6570000000000000),
    ("mixed_neg_2p600", 0x6570000000000000, 0xe570000000000000),
    ("real_2p-600", 0x1a70000000000000, 0x0000000000000000),
    ("neg_real_2p-600", 0x9a70000000000000, 0x0000000000000000),
    ("imag_2p-600", 0x0000000000000000, 0x1a70000000000000),
    ("neg_imag_2p-600", 0x0000000000000000, 0x9a70000000000000),
    ("mixed_2p-600", 0x1a70000000000000, 0x1a70000000000000),
    ("mixed_neg_2p-600", 0x1a70000000000000, 0x9a70000000000000),
    ("real_2p511", 0x5fe0000000000000, 0x0000000000000000),
    ("neg_real_2p511", 0xdfe0000000000000, 0x0000000000000000),
    ("imag_2p511", 0x0000000000000000, 0x5fe0000000000000),
    ("neg_imag_2p511", 0x0000000000000000, 0xdfe0000000000000),
    ("mixed_2p511", 0x5fe0000000000000, 0x5fe0000000000000),
    ("mixed_neg_2p511", 0x5fe0000000000000, 0xdfe0000000000000),
    ("real_2p512", 0x5ff0000000000000, 0x0000000000000000),
    ("neg_real_2p512", 0xdff0000000000000, 0x0000000000000000),
    ("imag_2p512", 0x0000000000000000, 0x5ff0000000000000),
    ("neg_imag_2p512", 0x0000000000000000, 0xdff0000000000000),
    ("mixed_2p512", 0x5ff0000000000000, 0x5ff0000000000000),
    ("mixed_neg_2p512", 0x5ff0000000000000, 0xdff0000000000000),
    ("real_2p-511", 0x2000000000000000, 0x0000000000000000),
    ("neg_real_2p-511", 0xa000000000000000, 0x0000000000000000),
    ("imag_2p-511", 0x0000000000000000, 0x2000000000000000),
    ("neg_imag_2p-511", 0x0000000000000000, 0xa000000000000000),
    ("mixed_2p-511", 0x2000000000000000, 0x2000000000000000),
    ("mixed_neg_2p-511", 0x2000000000000000, 0xa000000000000000),
    ("real_2p-512", 0x1ff0000000000000, 0x0000000000000000),
    ("neg_real_2p-512", 0x9ff0000000000000, 0x0000000000000000),
    ("imag_2p-512", 0x0000000000000000, 0x1ff0000000000000),
    ("neg_imag_2p-512", 0x0000000000000000, 0x9ff0000000000000),
    ("mixed_2p-512", 0x1ff0000000000000, 0x1ff0000000000000),
    ("mixed_neg_2p-512", 0x1ff0000000000000, 0x9ff0000000000000),
    ("real_2p1022", 0x7fd0000000000000, 0x0000000000000000),
    ("neg_real_2p1022", 0xffd0000000000000, 0x0000000000000000),
    ("imag_2p1022", 0x0000000000000000, 0x7fd0000000000000),
    ("neg_imag_2p1022", 0x0000000000000000, 0xffd0000000000000),
    ("mixed_2p1022", 0x7fd0000000000000, 0x7fd0000000000000),
    ("mixed_neg_2p1022", 0x7fd0000000000000, 0xffd0000000000000),
    ("real_2p-1022", 0x0010000000000000, 0x0000000000000000),
    ("neg_real_2p-1022", 0x8010000000000000, 0x0000000000000000),
    ("imag_2p-1022", 0x0000000000000000, 0x0010000000000000),
    ("neg_imag_2p-1022", 0x0000000000000000, 0x8010000000000000),
    ("mixed_2p-1022", 0x0010000000000000, 0x0010000000000000),
    ("mixed_neg_2p-1022", 0x0010000000000000, 0x8010000000000000),
    ("real_2p-1030_subnormal", 0x0000100000000000, 0x0000000000000000),
    ("neg_real_2p-1030_subnormal", 0x8000100000000000, 0x0000000000000000),
    ("imag_2p-1030_subnormal", 0x0000000000000000, 0x0000100000000000),
    ("neg_imag_2p-1030_subnormal", 0x0000000000000000, 0x8000100000000000),
    ("mixed_2p-1030_subnormal", 0x0000100000000000, 0x0000100000000000),
    ("mixed_neg_2p-1030_subnormal", 0x0000100000000000, 0x8000100000000000),
    ("real_min_subnormal", 0x0000000000000001, 0x0000000000000000),
    ("neg_real_min_subnormal", 0x8000000000000001, 0x0000000000000000),
    ("imag_min_subnormal", 0x0000000000000000, 0x0000000000000001),
    ("neg_imag_min_subnormal", 0x0000000000000000, 0x8000000000000001),
    ("mixed_min_subnormal", 0x0000000000000001, 0x0000000000000001),
    ("mixed_neg_min_subnormal", 0x0000000000000001, 0x8000000000000001),
    ("random_0_e-600", 0x9a74cfe900c1a500, 0x9a75f550b8be5d74),
    ("random_1_e0", 0xbfcb94ada8d8b6e0, 0xbff08b1b9c37253a),
    ("random_2_e600", 0x657c7fa94e666c6e, 0x6544c9ff7ba6d240),
    ("random_3_e0", 0x3fc2bf85a17ded20, 0xbfe06c0e79448bf4),
    ("random_4_e300", 0x52a01bf2feff8c6c, 0x5287d2c1929fd8e0),
    ("random_5_e0", 0xbffc3a4d108260f2, 0x3ff5a5c5b0fc374a),
    ("random_6_e-511", 0x200d5c421858e204, 0x2006d56f3f5eeab2),
    ("random_7_e300", 0xd2810502014008b0, 0xd2b3ddc653ace686),
    ("random_8_e600", 0xe55606afbfb70768, 0xe574ebdb893c703e),
    ("random_9_e511", 0xdfc981ba11426db0, 0x5feef280c2a756ca),
    ("random_10_e300", 0x52b1c2e360931d74, 0xd2bf46f5ef83e43e),
    ("random_11_e-511", 0x1fed0f254120eaf0, 0x1ff0c4746b0cbf5c),
    ("random_12_e0", 0x3ffb49eeeb5ae988, 0x3fea9ba1e943e740),
    ("random_13_e0", 0xbffd5f2889d72b10, 0xbfdfba02e2a09900),
    ("random_14_e511", 0xdfd28fe66fe6bccc, 0xdfe0ff9b1aefc19e),
    ("random_15_e-300", 0x2d23f03b405fa458, 0xad3f8a713e322fda),
    ("random_16_e10", 0xc07354f7b5ea1380, 0xc0751584d374f818),
    ("random_17_e10", 0x4073c665ae974b80, 0xc09b7d4e3904e454),
    ("random_18_e0", 0xbfe4c5267265c76c, 0xbfe70431b0e7be50),
    ("random_19_e-300", 0x2d30cb8128873cee, 0xad0505aa1c5793d0),
    ("random_20_e300", 0xd2b44f8b1890dcc4, 0x52a91630985e2bbc),
    ("random_21_e-600", 0x9a7bf20fc8f219bc, 0x1a59edd0092c4258),
    ("random_22_e-600", 0x9a74c1b6375593c0, 0x1a66e4e2bd2e9f20),
    ("random_23_e300", 0x52ab993847b2762c, 0x52bc1f9afaa9a1ec),
    ("random_24_e10", 0x408a00f8294a51fc, 0xc070aa570cd9fe98),
    ("random_25_e0", 0xbfc051c84f558ca0, 0xbfff687b59b97728),
    ("random_26_e0", 0xbff0123c8364d6e6, 0x3ff3db1f0ce9c3c8),
    ("random_27_e10", 0xc08213a901e2b644, 0xc09485872822f1e4),
    ("random_28_e-511", 0xa00456974153cb2a, 0xa005069789b3138e),
    ("random_29_e-511", 0xa00edc55e1bf4ff6, 0xa0021486827ec8ae),
    ("random_30_e511", 0x5fd8b1c735c96f58, 0xdfe33825d5430a96),
    ("random_31_e-511", 0x200a44509566a184, 0xa00bbaadf7637ea2),
    ("random_32_e10", 0x407f781e13b9c4a8, 0x40849787a41ddfac),
    ("random_33_e10", 0xc08e3dd91c91618c, 0xc0926f6959a27e18),
    ("random_34_e300", 0xd2a98e853409012c, 0xd290910ae59cb0e8),
    ("random_35_e300", 0xd28ff8a0d8aa1ef0, 0x52b47153aae5d706),
    ("random_36_e-600", 0x9a63837bf45fa1a0, 0x9a6620972a9166b4),
    ("random_37_e-600", 0x1a50cc6d3d0cd928, 0x1a7a0a0d622d33b2),
    ("random_38_e-511", 0xa0092d6a0d9edfb4, 0x200a5edbb6bad690),
    ("random_39_e10", 0x407d64fd1ea24430, 0xc05b8459a2ddd700),
    ("random_40_e300", 0xd2b6ef59fed7ad36, 0xd29c11f84d0ae980),
    ("random_41_e0", 0x3fdab7db2f18f5f0, 0x3fe7695b4fa62c2c),
    ("random_42_e-600", 0x9a7235c7e978dc70, 0x1a68f48459e3fd94),
    ("random_43_e-600", 0x1a61626e3c285444, 0x1a6e14b9ebe0cb80),
    ("random_44_e-511", 0xa000d7621b35fa04, 0xa00d0f8786814440),
    ("random_45_e-600", 0x9a7aa2c4c1447966, 0x1a78f77de9566642),
    ("random_46_e-600", 0x9a7120e7e0652758, 0x9a76bc32c73fa850),
    ("random_47_e10", 0xc09f91984fe2ab34, 0x4068752648f1b8a0),
    ("random_48_e-300", 0xad0421f184c1cb60, 0x2d1a2098916e3c40),
    ("random_49_e511", 0x5fb7e2bb97b0d290, 0xdfe155c918eb7880),
    ("random_50_e300", 0x52b609d8cba7d96c, 0x52b99d2b4549eac4),
    ("random_51_e511", 0x5fe33067d7ba564e, 0x5fe6f0a4056dfbb2),
    ("random_52_e600", 0xe574365e31a7cc0e, 0x653ac29f1f7d0e20),
    ("random_53_e600", 0xe5793b0fecf04826, 0x65711ef0f61433e2),
    ("random_54_e-511", 0x200bbdde16fac6a6, 0x200545904fefd94e),
    ("random_55_e0", 0x3fdeee3dc56012a8, 0x3ff0f58a4774e290),
    ("random_56_e600", 0xe576087191ebfe18, 0xe570dc2b12a86340),
    ("random_57_e-300", 0xad31e8353f823c54, 0xad0bf0b6f187a630),
    ("random_58_e-300", 0x2d2efcfae7023a00, 0x2d312fb54a27ce96),
    ("random_59_e0", 0xbfe8c28834b5b1a8, 0xbffa680e609c1ba6),
]

for (label, re_bits, im_bits) in CASES
    re = frombits(re_bits)
    im = frombits(im_bits)
    w = ComplexF64(re, im)
    r = inv(w)
    println(
        label, " ",
        hex16(re_bits), " ", hex16(im_bits), " ",
        hex16(bits(real(r))), " ", hex16(bits(imag(r))),
    )
end

# ---------------------------------------------------------------------------
# ComplexF32: `inv(z::Complex{Float32})` (`base/complex.jl:473`), ported as
# `julia_complex32_reciprocal_wide` (`tenet/src/typed.rs`). This path widens
# to `ComplexF64` before dividing, so — unlike the `ComplexF64` path above —
# it never reaches the scaled/`robust_cinv` fallback for any finite `f32`
# input; these cases exist to pin the widen/narrow port itself (including
# its infinite-input branch) bitwise, not to probe an overflow boundary.
# ---------------------------------------------------------------------------

bits32(x::Float32) = reinterpret(UInt32, x)
frombits32(u::UInt32) = reinterpret(Float32, u)
hex8(u::UInt32) = string(u; base=16, pad=8)

# (label, re bits, im bits) — large, small-normal, subnormal, and infinite
# f32 magnitudes, purely real/imaginary and mixed.
const CASES32 = [
    ("c32_large_real", 0x7f000000, 0x00000000),
    ("c32_large_imag", 0x00000000, 0x7f000000),
    ("c32_large_mixed", 0x7f000000, 0xff000000),
    ("c32_small_normal_real", 0x00800000, 0x00000000),
    ("c32_small_normal_imag", 0x00000000, 0x00800000),
    ("c32_small_normal_mixed", 0x00800000, 0x80800000),
    ("c32_subnormal_real", 0x00000010, 0x00000000),
    ("c32_subnormal_imag", 0x00000000, 0x00000010),
    ("c32_subnormal_min_mixed", 0x00000001, 0x80000001),
    ("c32_inf_re_pos_im_zero", 0x7f800000, 0x00000000),
    ("c32_inf_re_neg_im_zero", 0xff800000, 0x00000000),
    ("c32_re_zero_inf_im_pos", 0x00000000, 0x7f800000),
    ("c32_re_zero_inf_im_neg", 0x00000000, 0xff800000),
    ("c32_inf_re_pos_inf_im_pos", 0x7f800000, 0x7f800000),
    ("c32_inf_re_pos_finite_im", 0x7f800000, 0x40400000),
    ("c32_finite_re_inf_im_pos", 0x40400000, 0x7f800000),
]

for (label, re_bits, im_bits) in CASES32
    re = frombits32(re_bits)
    im = frombits32(im_bits)
    w = ComplexF32(re, im)
    r = inv(w)
    println(
        label, " ",
        hex8(re_bits), " ", hex8(im_bits), " ",
        hex8(bits32(real(r))), " ", hex8(bits32(imag(r))),
    )
end
