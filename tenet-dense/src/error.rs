use core::fmt;

use crate::{DenseBackend, DenseDType};

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DenseError {
    RankMismatch {
        shape: usize,
        strides: usize,
    },
    ElementCountOverflow,
    StrideOverflow {
        value: usize,
    },
    OffsetOverflow {
        value: usize,
    },
    OutOfBounds,
    Unsupported {
        op: &'static str,
        message: String,
    },
    DTypeMismatch {
        op: &'static str,
        expected: DenseDType,
        actual: DenseDType,
    },
    ShapeMismatch {
        op: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    NumericalFailure {
        backend: DenseBackend,
        op: &'static str,
        message: String,
    },
    Backend {
        backend: DenseBackend,
        op: &'static str,
        message: String,
    },
}

impl fmt::Display for DenseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RankMismatch { shape, strides } => {
                write!(
                    f,
                    "rank mismatch: shape rank {shape}, strides rank {strides}"
                )
            }
            Self::ElementCountOverflow => write!(f, "dense view element count overflow"),
            Self::StrideOverflow { value } => {
                write!(f, "dense view stride {value} does not fit in isize")
            }
            Self::OffsetOverflow { value } => {
                write!(f, "dense view offset {value} does not fit in isize")
            }
            Self::OutOfBounds => write!(f, "dense view accesses outside the buffer"),
            Self::Unsupported { op, message } => {
                write!(f, "unsupported dense operation {op}: {message}")
            }
            Self::DTypeMismatch {
                op,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "dense dtype mismatch in {op}: expected {expected:?}, got {actual:?}"
                )
            }
            Self::ShapeMismatch {
                op,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "dense shape mismatch in {op}: expected {expected:?}, got {actual:?}"
                )
            }
            Self::NumericalFailure {
                backend,
                op,
                message,
            } => {
                write!(f, "{backend:?} numerical failure in {op}: {message}")
            }
            Self::Backend {
                backend,
                op,
                message,
            } => {
                write!(f, "{backend:?} backend error in {op}: {message}")
            }
        }
    }
}

impl std::error::Error for DenseError {}
