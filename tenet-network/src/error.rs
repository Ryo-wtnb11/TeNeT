use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractError {
    EmptyEquation,
    EmptyInput,
    UnsupportedEllipsis,
    InvalidArrow,
    InvalidLabel(String),
    DuplicateOutputLabel(String),
    UnknownOutputLabel(String),
    TensorCountMismatch {
        expected: usize,
        actual: usize,
    },
    RankMismatch {
        tensor: usize,
        expected: usize,
        actual: usize,
    },
    DimensionMismatch {
        label: String,
        expected: usize,
        actual: usize,
    },
    InvalidTensorId {
        tensor: usize,
        tensor_count: usize,
    },
    InvalidBlockStructure(String),
    InvalidContractionPlan(String),
    UnsupportedExecution(String),
    TensorExecution(String),
    NotEnoughTensors,
    /// A label occurs more than once WITHIN a single operand (a diagonal /
    /// trace-on-one-tensor, e.g. `aa->a`). Not supported by the pairwise executor.
    UnsupportedDiagonal {
        label: String,
        tensor: usize,
    },
    /// A label is shared by more than two operands (a hyperedge, e.g.
    /// `a,a,a->`). The pairwise executor contracts a label across exactly two
    /// operands; >2 is unsupported.
    UnsupportedHyperedge {
        label: String,
        operand_count: usize,
    },
    /// An OUTPUT label is carried by more than one input operand (a
    /// batch/hadamard index, e.g. `ab,ab->ab`). Not supported.
    UnsupportedBatchLabel {
        label: String,
        operand_count: usize,
    },
    /// A contracted (non-output) label occurs on only ONE operand, so it would
    /// need a single-operand reduction/sum (e.g. `a->` or `ab->a`). Not supported.
    UnsupportedReduction {
        label: String,
    },
}

pub type Result<T> = std::result::Result<T, ContractError>;

impl Display for ContractError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ContractError::EmptyEquation => write!(f, "empty contraction equation"),
            ContractError::EmptyInput => write!(f, "contraction equation has an empty input"),
            ContractError::UnsupportedEllipsis => write!(f, "ellipsis is not supported yet"),
            ContractError::InvalidArrow => write!(f, "equation must contain at most one `->`"),
            ContractError::InvalidLabel(label) => write!(f, "invalid temporary label `{label}`"),
            ContractError::DuplicateOutputLabel(label) => {
                write!(f, "duplicate output label `{label}`")
            }
            ContractError::UnknownOutputLabel(label) => {
                write!(f, "output label `{label}` does not occur in inputs")
            }
            ContractError::TensorCountMismatch { expected, actual } => {
                write!(f, "expected {expected} tensor infos, got {actual}")
            }
            ContractError::RankMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "rank mismatch for tensor {tensor}: expected {expected}, got {actual}"
            ),
            ContractError::DimensionMismatch {
                label,
                expected,
                actual,
            } => write!(
                f,
                "dimension mismatch for label `{label}`: expected {expected}, got {actual}"
            ),
            ContractError::InvalidTensorId {
                tensor,
                tensor_count,
            } => write!(
                f,
                "invalid tensor id {tensor}; network has {tensor_count} tensors"
            ),
            ContractError::InvalidBlockStructure(message) => {
                write!(f, "invalid block-sparse tensor info: {message}")
            }
            ContractError::InvalidContractionPlan(message) => {
                write!(f, "invalid contraction plan: {message}")
            }
            ContractError::UnsupportedExecution(message) => {
                write!(f, "unsupported contraction execution: {message}")
            }
            ContractError::TensorExecution(message) => {
                write!(f, "tensor execution failed: {message}")
            }
            ContractError::NotEnoughTensors => write!(f, "need at least two active tensors"),
            ContractError::UnsupportedDiagonal { label, tensor } => write!(
                f,
                "einsum: repeated label `{label}` within one operand (tensor {tensor}) \
                 is a diagonal/trace — not supported"
            ),
            ContractError::UnsupportedHyperedge {
                label,
                operand_count,
            } => write!(
                f,
                "einsum: label `{label}` appears on {operand_count} operands (>2, a \
                 hyperedge) — not supported"
            ),
            ContractError::UnsupportedBatchLabel {
                label,
                operand_count,
            } => write!(
                f,
                "einsum: output label `{label}` is shared by {operand_count} inputs \
                 (batch/hadamard) — not supported"
            ),
            ContractError::UnsupportedReduction { label } => write!(
                f,
                "einsum: contracted label `{label}` occurs on a single operand \
                 (single-operand reduction) — not supported"
            ),
        }
    }
}

impl std::error::Error for ContractError {}
