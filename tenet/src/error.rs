//! User-layer error type.

use std::fmt;

use tenet_core::{CoreError, FusionAlgebraError};
use tenet_tensors::OperationError;

use crate::tensor::Dtype;

/// Error produced by the user-layer [`crate::prelude::Tensor`] /
/// [`crate::prelude::Space`] / [`crate::prelude::Runtime`] API.
///
/// Expert-layer errors ([`CoreError`], [`OperationError`]) are passed through
/// unchanged; the remaining variants report user-level misuse (mixing rules
/// or mixing runtimes).
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// Structural error bubbled up from `tenet-core` (boxed: the expert
    /// error types are large, and boxing keeps `Result<_, Error>` returns
    /// small — the `clippy::result_large_err` fix).
    Core(Box<CoreError>),
    /// Execution error bubbled up from the expert operation layer (boxed,
    /// same reason).
    Operation(Box<OperationError>),
    /// A finite encoded fusion algebra cannot represent the requested
    /// mathematical dual or fusion output.
    FusionAlgebra(Box<FusionAlgebraError>),
    /// The operands carry different fusion rules (e.g. U1 vs Z2).
    RuleMismatch,
    /// The operands belong to different [`crate::prelude::Runtime`]s.
    RuntimeMismatch,
    /// An operation that requires equal scalar types received f64 and c64.
    /// Mixed-dtype operations document their conversion behavior separately.
    DtypeMismatch,
    /// A scalar cannot be represented exactly in the destination dtype.
    InexactScalarConversion {
        /// Stable public operation name, such as `Tensor::absorb`.
        operation: &'static str,
        /// Scalar dtype being converted.
        from: Dtype,
        /// Required destination dtype.
        to: Dtype,
    },
    /// The operands live on different placements (host vs device, or
    /// different devices); transfer explicitly with `to_cuda()` / `to_host()`
    /// first.
    PlacementMismatch,
    /// The operation has no device implementation yet; the message says
    /// which. Device tensors never fall back to host execution silently —
    /// move the tensor explicitly with `to_host()`.
    UnsupportedOnDevice(String),
    /// The operation is part of the public API but is not implemented for the
    /// operand's fusion rule yet.
    UnsupportedForRule {
        /// Stable public operation name, such as `Tensor::eigh_full`.
        operation: &'static str,
        /// User-facing fusion-rule name, such as `SU(3)`.
        rule: &'static str,
    },
    /// Invalid user input (axes, sectors, spaces); the message says what.
    InvalidArgument(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(err) => write!(f, "core error: {err}"),
            Self::Operation(err) => write!(f, "operation error: {err}"),
            Self::FusionAlgebra(err) => write!(f, "fusion algebra error: {err}"),
            Self::RuleMismatch => write!(f, "operands use different fusion rules"),
            Self::RuntimeMismatch => write!(f, "operands belong to different runtimes"),
            Self::DtypeMismatch => write!(f, "operands store different scalar types"),
            Self::InexactScalarConversion {
                operation,
                from,
                to,
            } => write!(
                f,
                "{operation} cannot convert a scalar exactly from {from:?} to {to:?}"
            ),
            Self::PlacementMismatch => write!(
                f,
                "operands live on different placements (transfer explicitly first)"
            ),
            Self::UnsupportedOnDevice(message) => {
                write!(f, "unsupported on device: {message}")
            }
            Self::UnsupportedForRule { operation, rule } => {
                write!(f, "{operation} is not supported for {rule}")
            }
            Self::InvalidArgument(message) => write!(f, "invalid argument: {message}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Core(err) => Some(err.as_ref()),
            Self::Operation(err) => Some(err.as_ref()),
            Self::FusionAlgebra(err) => Some(err.as_ref()),
            _ => None,
        }
    }
}

impl From<CoreError> for Error {
    fn from(err: CoreError) -> Self {
        Self::Core(Box::new(err))
    }
}

impl From<OperationError> for Error {
    fn from(err: OperationError) -> Self {
        match err {
            OperationError::FusionAlgebra(cause) => Self::FusionAlgebra(cause),
            other => Self::Operation(Box::new(other)),
        }
    }
}

impl From<FusionAlgebraError> for Error {
    fn from(err: FusionAlgebraError) -> Self {
        Self::FusionAlgebra(Box::new(err))
    }
}
