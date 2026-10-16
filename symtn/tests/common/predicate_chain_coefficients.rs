// `ChainCoefficient` for the four payload dtypes, for integration tests that
// include `predicate_chains.rs`.

impl ChainCoefficient for f64 {
    fn real(value: f64) -> Self {
        value
    }
}

impl ChainCoefficient for f32 {
    fn real(value: f64) -> Self {
        value as f32
    }
}

impl ChainCoefficient for num_complex::Complex64 {
    fn real(value: f64) -> Self {
        num_complex::Complex64::new(value, 0.0)
    }
}

impl ChainCoefficient for num_complex::Complex32 {
    fn real(value: f64) -> Self {
        num_complex::Complex32::new(value as f32, 0.0)
    }
}
