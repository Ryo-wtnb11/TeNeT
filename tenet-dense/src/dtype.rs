#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DenseDType {
    F32,
    F64,
    I32,
    I64,
    Bool,
    C32,
    C64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DenseBackend {
    /// Host CPU dense executor (tenferro-cpu).
    Tenferro,
    /// CUDA device dense boundary (tenferro-gpu / cuSOLVER / cuBLAS).
    Cuda,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DensePlacement {
    Host,
}

#[cfg(feature = "tenferro")]
pub(crate) fn dense_dtype_from_tenferro(dtype: tenferro_tensor::DType) -> DenseDType {
    match dtype {
        tenferro_tensor::DType::F32 => DenseDType::F32,
        tenferro_tensor::DType::F64 => DenseDType::F64,
        tenferro_tensor::DType::I32 => DenseDType::I32,
        tenferro_tensor::DType::I64 => DenseDType::I64,
        tenferro_tensor::DType::Bool => DenseDType::Bool,
        tenferro_tensor::DType::C32 => DenseDType::C32,
        tenferro_tensor::DType::C64 => DenseDType::C64,
    }
}
