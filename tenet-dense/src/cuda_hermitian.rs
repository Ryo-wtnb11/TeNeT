//! The device Hermitian admission rule, as pure host arithmetic.
//!
//! Compiled (and tested) without the `cuda` feature, like `cuda_region`, so
//! CI — which merely `cargo check`s that feature — still executes the rule
//! that decides whether a device block is offered to `eigh`.

/// How many machine epsilons of the payload's **real lane** the relative
/// anti-Hermitian residual may reach.
///
/// The constant is the host twin's, not a device-specific one: the CPU rule is
/// `||(A - A†)/2||_F <= 64 * eps(R) * ||A||_F` with
/// `R::relative_tolerance() == 64 * R::EPSILON` for `R` in `{f32, f64}`
/// (`tenet-matrixalgebra/src/factorize.rs:6796` `normwise_hermitian`, `:177`
/// and `:187`). TensorKit agrees on the *shape*, not on the number: its
/// `ishermitian` (`TensorKit.jl` `cfaa073e`
/// `src/factorizations/factorizations.jl:67`) forwards each block to
/// MatrixAlgebraKit 0.6.9 `ishermitian`
/// (`src/common/matrixproperties.jl:72`), which is exact equality by default
/// and, when a tolerance is asked for, takes it from the payload's own
/// epsilon — `default_hermitian_tol(A) = eps(norm(A, Inf))^(3/4)`
/// (`src/common/defaults.jl:43`), used by `strided_ishermitian_approx`
/// (`matrixproperties.jl:150`). Neither reference ever compares a
/// single-precision block against a double-precision epsilon, which is
/// exactly what the device rule did before C1: `64 * eps(f64) = 1.4e-14`
/// against the `64 * eps(f32) = 7.6e-6` an `f32` block is entitled to, a
/// factor of ~5e8 that rejects every genuinely rounded `f32` block.
pub(crate) const HERMITIAN_TOLERANCE_EPSILONS: f64 = 64.0;

/// Decides `0.5 * ||A - A†||_F <= tolerance * ||A||_F` from the *scaled*
/// device reductions.
///
/// Both norms arrive factored as `scale * sqrt(sum_of_squares)` of a
/// copy normalized by an exact power of two near its maximum
/// ([`power_of_two_normalizer`]), which is what keeps the relative decision
/// stable when either norm would overflow or underflow if formed directly —
/// the residual is formed from the normalized input, so the input normalizer
/// cancels out of both sides and only the residual's `residual_scale`
/// (the reciprocal of its own normalizer) remains. A non-finite or negative part
/// rejects: it can only come from a non-finite block.
///
/// `relative_tolerance` is `HERMITIAN_TOLERANCE_EPSILONS * eps(real(D))` for
/// the payload under test; it is a parameter so this rule is one authority for
/// all four payload dtypes rather than four copies.
pub(crate) fn scaled_hermitian_residual_accepts(
    input_ss: f64,
    residual_scale: f64,
    residual_ss: f64,
    relative_tolerance: f64,
) -> bool {
    input_ss.is_finite()
        && input_ss >= 0.0
        && residual_scale.is_finite()
        && residual_scale >= 0.0
        && residual_ss.is_finite()
        && residual_ss >= 0.0
        && 0.5 * residual_scale * residual_ss.sqrt() <= relative_tolerance * input_ss.sqrt()
}

/// The exact power of two `2^-k`, `k = floor(log2(scale))`, that brings a
/// finite positive `scale` of a real lane with `max_exp` (its `MAX_EXP`)
/// into `[1, 2)`.
///
/// Why not divide by `scale` itself: Tenferro's complex-by-real division
/// promotes the divisor to `Complex(s, 0)` and forms `s * s`, which leaves
/// the lane once `|s|` passes the square root of its range (tenferro-rs#1922),
/// while its complex-by-real multiplication is componentwise. Multiplying by
/// a power of two is also exact wherever the product stays normal.
///
/// `k` is read from the exponent bits, never from a float `log2`. It is
/// clamped so `2^-k` is a *normal* number of the lane: below
/// `2^-(MAX_EXP - 1)` (only a subnormal `scale`) `2^-k` would overflow, and
/// the clamped product still has a normal maximum (`>= 2^-22` in `f32`,
/// `>= 2^-51` in `f64`); at `2^(MAX_EXP - 1)` and above `2^-k` would be
/// subnormal, which a flush-to-zero kernel may zero, and the clamped product
/// is merely in `[2, 4)`.
pub(crate) fn power_of_two_normalizer(scale: f64, max_exp: i32) -> f64 {
    const MANTISSA_BITS: u32 = 52;
    const BIAS: i32 = 1023;
    let bits = scale.to_bits();
    let biased = ((bits >> MANTISSA_BITS) & 0x7ff) as i32;
    let floor_log2 = if biased == 0 {
        // An `f64` subnormal is `mantissa * 2^-1074`.
        let mantissa = bits & ((1 << MANTISSA_BITS) - 1);
        63 - mantissa.leading_zeros() as i32 - 1074
    } else {
        biased - BIAS
    };
    let k = floor_log2.clamp(-(max_exp - 1), max_exp - 2);
    f64::from_bits(((BIAS - k) as u64) << MANTISSA_BITS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tolerance(epsilon: f64) -> f64 {
        HERMITIAN_TOLERANCE_EPSILONS * epsilon
    }

    #[test]
    fn the_rule_uses_the_shared_half_residual_threshold() {
        let input_ss: f64 = 2.0;
        let tolerance = tolerance(f64::EPSILON);
        let threshold = tolerance * input_ss.sqrt();
        assert!(scaled_hermitian_residual_accepts(
            input_ss,
            2.0 * threshold * 0.99,
            1.0,
            tolerance,
        ));
        assert!(!scaled_hermitian_residual_accepts(
            input_ss,
            2.0 * threshold * 1.01,
            1.0,
            tolerance,
        ));
        assert!(!scaled_hermitian_residual_accepts(
            input_ss,
            f64::NAN,
            1.0,
            tolerance
        ));
        assert!(!scaled_hermitian_residual_accepts(
            input_ss,
            1.0,
            f64::INFINITY,
            tolerance,
        ));
    }

    /// The C1 hazard, as host arithmetic: a residual of a few `f32` roundings
    /// relative to the input norm is accepted by the single-precision
    /// tolerance and rejected by the double-precision one. Both are expressed
    /// in epsilons of their own lane, so nothing here is a platform-dependent
    /// absolute constant.
    #[test]
    fn a_single_precision_rounding_residual_needs_the_single_precision_lane() {
        let input_ss: f64 = 1.0;
        // Half the anti-Hermitian residual sits at 8 f32 epsilons of the input
        // norm: well inside `64 * eps(f32)`, and ~4.7e8 times outside
        // `64 * eps(f64)`.
        let residual_scale = 2.0 * 8.0 * f64::from(f32::EPSILON);
        assert!(scaled_hermitian_residual_accepts(
            input_ss,
            residual_scale,
            1.0,
            tolerance(f64::from(f32::EPSILON)),
        ));
        assert!(!scaled_hermitian_residual_accepts(
            input_ss,
            residual_scale,
            1.0,
            tolerance(f64::EPSILON),
        ));

        // A clearly non-Hermitian block is rejected in either lane: the same
        // residual grown past the single-precision threshold.
        let asymmetric = 2.0 * 256.0 * f64::from(f32::EPSILON);
        assert!(!scaled_hermitian_residual_accepts(
            input_ss,
            asymmetric,
            1.0,
            tolerance(f64::from(f32::EPSILON)),
        ));
    }

    /// The `f64` decisions the rule made before the tolerance became a
    /// parameter are unchanged: the threshold is still exactly
    /// `64 * eps(f64) * ||A||_F` on the half-residual.
    #[test]
    fn the_double_precision_decisions_are_unchanged() {
        let tolerance = tolerance(f64::EPSILON);
        for &input_ss in &[1.0_f64, 2.0, 1e-30, 1e30] {
            let threshold = 2.0 * tolerance * input_ss.sqrt();
            assert!(scaled_hermitian_residual_accepts(
                input_ss,
                threshold * 0.99,
                1.0,
                tolerance
            ));
            assert!(!scaled_hermitian_residual_accepts(
                input_ss,
                threshold * 1.01,
                1.0,
                tolerance
            ));
        }
    }

    /// Every finite positive scale of either lane lands in `[1, 2)` except
    /// where the clamp is documented to act, and the multiplier is always a
    /// normal number of the lane, so narrowing and flush-to-zero leave it alone.
    #[test]
    fn the_power_of_two_normalizer_is_exact_across_each_lane() {
        fn check(scale: f64, max_exp: i32, min_positive: f64, max_power: f64) {
            let normalizer = power_of_two_normalizer(scale, max_exp);
            assert!(
                (min_positive..=max_power).contains(&normalizer),
                "{scale:e}: {normalizer:e} is not a normal lane power of two"
            );
            assert_eq!(normalizer.to_bits() & ((1 << 52) - 1), 0, "{scale:e}");
            let scaled = scale * normalizer;
            if scale < 2.0 / max_power {
                assert_eq!(normalizer, max_power, "{scale:e}");
            } else if scale >= max_power {
                assert!((2.0..4.0).contains(&scaled), "{scale:e}: {scaled:e}");
            } else {
                assert!((1.0..2.0).contains(&scaled), "{scale:e}: {scaled:e}");
            }
        }
        let f32_lane = |scale: f64| {
            check(
                scale,
                f32::MAX_EXP,
                f64::from(f32::MIN_POSITIVE),
                2f64.powi(127),
            )
        };
        for scale in [
            f32::from_bits(1),
            f32::from_bits(0x0000_1234),
            f32::MIN_POSITIVE,
            f32::MIN_POSITIVE * 1.5,
            0.75,
            1.0,
            1.999_999_9,
            2.0f32.powi(124) * 5.0,
            2.0f32.powi(127),
            f32::MAX,
        ] {
            f32_lane(f64::from(scale));
        }
        let f64_lane = |scale: f64| check(scale, f64::MAX_EXP, f64::MIN_POSITIVE, 2f64.powi(1023));
        for scale in [
            f64::from_bits(1),
            f64::from_bits(0x000f_ffff_ffff_ffff),
            f64::MIN_POSITIVE,
            3.0e-300,
            1.0,
            1.0 - f64::EPSILON / 2.0,
            2f64.powi(1020) * 5.0,
            2f64.powi(1023),
            f64::MAX,
        ] {
            f64_lane(scale);
        }
        // Subnormal lane scales are clamped yet still end in the lane's normal
        // range after the multiply.
        assert!(
            f64::from(f32::from_bits(1))
                * power_of_two_normalizer(f64::from(f32::from_bits(1)), f32::MAX_EXP)
                >= 2f64.powi(-22)
        );
        assert!(
            f64::from_bits(1) * power_of_two_normalizer(f64::from_bits(1), f64::MAX_EXP)
                >= 2f64.powi(-51)
        );
    }
}
