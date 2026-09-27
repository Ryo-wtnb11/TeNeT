use super::*;

/// The same fixture as the device test `hermitian_extremes_case`
/// (`tenet-dense/tests/cuda_scalar_dtypes.rs`): Hermitian by construction
/// and scaled by an exact power of two.
fn hermitian_block<D: FactorScalar>(n: usize, scale: f64) -> Vec<D> {
    (0..n * n)
        .map(|index| {
            let (row, col) = (index % n, index / n);
            let (low, high) = (row.min(col), row.max(col));
            let magnitude = 1.0 + 0.5 * ((low % 7) as f64) + 0.25 * ((high % 5) as f64);
            let imaginary = 0.5 + ((low % 3) as f64);
            let imaginary = match row.cmp(&col) {
                std::cmp::Ordering::Equal => 0.0,
                std::cmp::Ordering::Less => imaginary,
                std::cmp::Ordering::Greater => -imaginary,
            };
            D::from_complex64(Complex64::new(magnitude * scale, imaginary * scale))
        })
        .collect()
}

/// The host twin of the device rule decides at the device test's scales —
/// the lane's smallest normal and `2^(MAX_EXP - 4)` — as it does at 1.
fn extremes_case<D: FactorScalar>(tiny: f64, huge: f64) {
    let n = 4;
    for scale in [tiny, 1.0, huge] {
        let hermitian = hermitian_block::<D>(n, scale);
        assert!(hermitian_matrix_contents(&hermitian, n), "{scale:e}");
        let mut asymmetric = hermitian;
        asymmetric[1] =
            D::from_complex64(asymmetric[1].widen_complex() + Complex64::new(scale, 0.0));
        assert!(!hermitian_matrix_contents(&asymmetric, n), "{scale:e}");
    }
}

#[test]
fn the_host_rule_is_scale_invariant_across_the_normal_range() {
    let (f32_tiny, f32_huge) = (f64::from(f32::MIN_POSITIVE), 2f64.powi(124));
    let (f64_tiny, f64_huge) = (f64::MIN_POSITIVE, 2f64.powi(1020));
    extremes_case::<f32>(f32_tiny, f32_huge);
    extremes_case::<f64>(f64_tiny, f64_huge);
    extremes_case::<num_complex::Complex32>(f32_tiny, f32_huge);
    extremes_case::<Complex64>(f64_tiny, f64_huge);
}
