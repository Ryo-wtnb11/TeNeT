//! User-layer error type.

use std::fmt;

use tenet_core::{CoreError, FusionAlgebraError};
use tenet_matrixalgebra::TruncationError;
use tenet_tensors::OperationError;

use crate::typed::{BatchMemberRepresentation, SignatureField};

/// Error produced by the user-layer [`crate::prelude::TensorMap`] /
/// [`crate::prelude::GradedSpace`] / [`crate::prelude::Runtime`] API.
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
    /// The operands live on different placements (host vs device, or
    /// different devices); transfer explicitly with `to_cuda()` / `to_host()`
    /// first.
    PlacementMismatch,
    /// The operation has no device implementation yet; the message says
    /// which. Device tensors never fall back to host execution silently —
    /// move the tensor explicitly with `to_host()`.
    UnsupportedOnDevice(String),
    /// Invalid user input (axes, sectors, spaces); the message says what.
    InvalidArgument(String),
    /// A batch operand does not match one structure signature: `member` is
    /// the first differing member of a pack, or `None` when a whole stack
    /// differs from a prepared handle. `field` is the first differing
    /// determinant.
    BatchSignatureMismatch {
        /// The differing member of a pack; `None` for a whole stack.
        member: Option<usize>,
        /// The first differing determinant.
        field: SignatureField,
    },
    /// A batch member's payload is not an owned dense buffer.
    UnsupportedBatchMember {
        /// The rejected member.
        member: usize,
        /// Its payload representation.
        representation: BatchMemberRepresentation,
    },
    /// A member index is not below the stack's member count.
    BatchMemberOutOfRange {
        /// The requested member.
        member: usize,
        /// The stack's member count.
        len: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(err) => write!(f, "core error: {err}"),
            Self::Operation(err) => write!(f, "operation error: {err}"),
            Self::FusionAlgebra(err) => write!(f, "fusion algebra error: {err}"),
            Self::RuleMismatch => write!(f, "operands use different fusion rules"),
            Self::RuntimeMismatch => write!(f, "operands belong to different runtimes"),
            Self::PlacementMismatch => write!(
                f,
                "operands live on different placements (transfer explicitly first)"
            ),
            Self::UnsupportedOnDevice(message) => {
                write!(f, "unsupported on device: {message}")
            }
            Self::InvalidArgument(message) => write!(f, "invalid argument: {message}"),
            Self::BatchSignatureMismatch {
                member: Some(member),
                field,
            } => write!(f, "batch member {member} differs from member 0 in {field:?}"),
            Self::BatchSignatureMismatch {
                member: None,
                field,
            } => write!(f, "stack differs from the prepared handle in {field:?}"),
            Self::UnsupportedBatchMember {
                member,
                representation,
            } => write!(f, "batch member {member} is a {representation:?} payload; pack needs owned dense members"),
            Self::BatchMemberOutOfRange { member, len } => {
                write!(f, "member {member} is out of range for a stack of {len}")
            }
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

impl From<TruncationError> for Error {
    fn from(err: TruncationError) -> Self {
        OperationError::from(err).into()
    }
}

impl From<FusionAlgebraError> for Error {
    fn from(err: FusionAlgebraError) -> Self {
        Self::FusionAlgebra(Box::new(err))
    }
}
