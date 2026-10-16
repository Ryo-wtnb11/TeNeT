use num_complex::{Complex32, Complex64};

/// Dtype-erased GEMM scalar for the accumulate-form matmul seam
/// (`C = alpha * A * B + beta * C`). Mirrors the BLAS/cuTENSOR parameter
/// shape so backends can consume it without generics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DenseScalar {
    F32(f32),
    F64(f64),
    C32(Complex32),
    C64(Complex64),
}

impl DenseScalar {
    pub fn is_one(&self) -> bool {
        match self {
            Self::F32(value) => *value == 1.0,
            Self::F64(value) => *value == 1.0,
            Self::C32(value) => *value == Complex32::new(1.0, 0.0),
            Self::C64(value) => *value == Complex64::new(1.0, 0.0),
        }
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Self::F32(value) => *value == 0.0,
            Self::F64(value) => *value == 0.0,
            Self::C32(value) => *value == Complex32::new(0.0, 0.0),
            Self::C64(value) => *value == Complex64::new(0.0, 0.0),
        }
    }
}
