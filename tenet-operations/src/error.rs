use core::fmt;

use tenet_core::{BlockKey, BraidingStyleKind, CoreError, FusionStyleKind};
use tenet_dense::DenseError;

use crate::TreeTransformOperation;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OperationError {
    Core(CoreError),
    Dense(DenseError),
    BlockIndexOutOfBounds {
        tensor: &'static str,
        index: usize,
        count: usize,
    },
    BlockCountMismatch {
        dst: usize,
        src: usize,
    },
    CoefficientCountMismatch {
        expected: usize,
        actual: usize,
    },
    ContractAxisCountMismatch {
        lhs: usize,
        rhs: usize,
    },
    TraceAxisCountMismatch {
        lhs: usize,
        rhs: usize,
    },
    DuplicateTransformDestination {
        dst_block: usize,
    },
    ElementCountMismatch {
        expected: usize,
        actual: usize,
    },
    ElementCountOverflow,
    EmptyTransformBlock,
    ExpectedFusionTreeBlock {
        tensor: &'static str,
        index: usize,
    },
    ExpectedAllCodomainFusionTree {
        index: usize,
    },
    InvalidPermutation {
        axes: Vec<usize>,
        rank: usize,
    },
    InvalidAxisSet {
        tensor: &'static str,
        axes: Vec<usize>,
        rank: usize,
    },
    FusionTreeGroupMismatch {
        tensor: &'static str,
        index: usize,
    },
    RankMismatch {
        expected: usize,
        actual: usize,
    },
    StructureMismatch {
        tensor: &'static str,
    },
    StructureRankMismatch {
        expected: usize,
        actual: usize,
    },
    UnsupportedFusionStyle {
        operation: TreeTransformOperation,
        style: FusionStyleKind,
    },
    UnsupportedBraidingStyle {
        operation: TreeTransformOperation,
        style: BraidingStyleKind,
    },
    UnsupportedTreeTransformScope {
        operation: TreeTransformOperation,
        message: &'static str,
    },
    UnsupportedTensorContractScope {
        message: &'static str,
    },
    MissingBlockKey {
        key: BlockKey,
    },
    ShapeMismatch {
        dst: Vec<usize>,
        src: Vec<usize>,
    },
    StrideOverflow {
        value: usize,
    },
    OffsetOverflow {
        value: usize,
    },
    StridedKernel {
        message: String,
    },
}

impl fmt::Display for OperationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(err) => err.fmt(f),
            Self::Dense(err) => err.fmt(f),
            Self::BlockIndexOutOfBounds {
                tensor,
                index,
                count,
            } => {
                write!(
                    f,
                    "{tensor} block index {index} is out of bounds for {count} blocks"
                )
            }
            Self::BlockCountMismatch { dst, src } => {
                write!(f, "block count mismatch: dst {dst}, src {src}")
            }
            Self::CoefficientCountMismatch { expected, actual } => {
                write!(
                    f,
                    "coefficient count mismatch: expected {expected}, got {actual}"
                )
            }
            Self::ContractAxisCountMismatch { lhs, rhs } => {
                write!(f, "contracting axis count mismatch: lhs {lhs}, rhs {rhs}")
            }
            Self::TraceAxisCountMismatch { lhs, rhs } => {
                write!(f, "trace axis count mismatch: lhs {lhs}, rhs {rhs}")
            }
            Self::DuplicateTransformDestination { dst_block } => {
                write!(
                    f,
                    "tree transform destination block {dst_block} appears in more than one block"
                )
            }
            Self::ElementCountMismatch { expected, actual } => {
                write!(
                    f,
                    "element count mismatch: expected {expected}, got {actual}"
                )
            }
            Self::ElementCountOverflow => write!(f, "element count overflow"),
            Self::EmptyTransformBlock => {
                write!(f, "tree transform block has no source or destination")
            }
            Self::ExpectedFusionTreeBlock { tensor, index } => {
                write!(f, "{tensor} block {index} is not a fusion-tree block")
            }
            Self::ExpectedAllCodomainFusionTree { index } => {
                write!(
                    f,
                    "source fusion-tree block {index} is not an all-codomain tree"
                )
            }
            Self::InvalidPermutation { axes, rank } => {
                write!(f, "invalid axis permutation {axes:?} for rank {rank}")
            }
            Self::InvalidAxisSet { tensor, axes, rank } => {
                write!(f, "invalid {tensor} axis set {axes:?} for rank {rank}")
            }
            Self::FusionTreeGroupMismatch { tensor, index } => {
                write!(
                    f,
                    "{tensor} block {index} does not match the fusion-tree group"
                )
            }
            Self::RankMismatch { expected, actual } => {
                write!(f, "rank mismatch: expected {expected}, got {actual}")
            }
            Self::StructureMismatch { tensor } => {
                write!(
                    f,
                    "{tensor} tensor structure does not match compiled structure"
                )
            }
            Self::StructureRankMismatch { expected, actual } => {
                write!(
                    f,
                    "block structure rank mismatch: expected {expected}, got {actual}"
                )
            }
            Self::UnsupportedFusionStyle { operation, style } => {
                write!(
                    f,
                    "unsupported fusion style {style:?} for tree transform operation {operation:?}"
                )
            }
            Self::UnsupportedBraidingStyle { operation, style } => {
                write!(
                    f,
                    "unsupported braiding style {style:?} for tree transform operation {operation:?}"
                )
            }
            Self::UnsupportedTreeTransformScope { operation, message } => {
                write!(
                    f,
                    "unsupported tree transform scope for operation {operation:?}: {message}"
                )
            }
            Self::UnsupportedTensorContractScope { message } => {
                write!(f, "unsupported tensor contraction scope: {message}")
            }
            Self::MissingBlockKey { key } => {
                write!(f, "missing matching block for key {key:?}")
            }
            Self::ShapeMismatch { dst, src } => {
                write!(f, "shape mismatch: dst {dst:?}, src {src:?}")
            }
            Self::StrideOverflow { value } => {
                write!(f, "stride {value} does not fit in strided-rs isize")
            }
            Self::OffsetOverflow { value } => {
                write!(f, "offset {value} does not fit in strided-rs isize")
            }
            Self::StridedKernel { message } => write!(f, "strided kernel error: {message}"),
        }
    }
}

impl std::error::Error for OperationError {}

impl From<CoreError> for OperationError {
    fn from(value: CoreError) -> Self {
        Self::Core(value)
    }
}

impl OperationError {
    /// Wraps a core error without collapsing its context; shared with the
    /// matrix-algebra crate.
    pub fn from_core_preserving_context(value: CoreError) -> Self {
        match value {
            CoreError::MissingBlockKey { key } => Self::MissingBlockKey { key },
            other => Self::Core(other),
        }
    }
}
