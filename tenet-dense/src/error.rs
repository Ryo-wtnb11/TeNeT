use core::fmt;

use crate::DenseBackend;

#[derive(Clone, Debug, PartialEq, Eq)]
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
