//! User-layer error type.

use std::fmt;

use tenet_core::CoreError;
use tenet_tensors::OperationError;

/// Error produced by the user-layer [`crate::prelude::Tensor`] /
/// [`crate::prelude::Space`] / [`crate::prelude::Runtime`] API.
///
/// Expert-layer errors ([`CoreError`], [`OperationError`]) are passed through
/// unchanged; the remaining variants report user-level misuse (mixing rules
/// or mixing runtimes).
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// Structural error bubbled up from `tenet-core`.
    Core(CoreError),
    /// Execution error bubbled up from the expert operation layer.
    Operation(OperationError),
    /// The operands carry different fusion rules (e.g. U1 vs Z2).
    RuleMismatch,
    /// The operands belong to different [`crate::prelude::Runtime`]s.
    RuntimeMismatch,
    /// Invalid user input (axes, sectors, spaces); the message says what.
    InvalidArgument(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(err) => write!(f, "core error: {err}"),
            Self::Operation(err) => write!(f, "operation error: {err}"),
            Self::RuleMismatch => write!(f, "operands use different fusion rules"),
            Self::RuntimeMismatch => write!(f, "operands belong to different runtimes"),
            Self::InvalidArgument(message) => write!(f, "invalid argument: {message}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<CoreError> for Error {
    fn from(err: CoreError) -> Self {
        Self::Core(err)
    }
}

impl From<OperationError> for Error {
    fn from(err: OperationError) -> Self {
        Self::Operation(err)
    }
}
