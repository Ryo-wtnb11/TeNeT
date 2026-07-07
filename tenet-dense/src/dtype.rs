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
    Tenferro,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DensePlacement {
    Host,
}
