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
