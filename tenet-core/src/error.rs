#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CoreError {
    RankMismatch {
        shape: usize,
        strides: usize,
    },
    StructureRankMismatch {
        expected: usize,
        actual: usize,
    },
    FusionSpaceSplitMismatch {
        expected_nout: usize,
        expected_nin: usize,
        actual_nout: usize,
        actual_nin: usize,
    },
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidBraidIndex {
        index: usize,
        rank: usize,
    },
    InvalidPermutation {
        permutation: Vec<usize>,
        rank: usize,
    },
    UnsupportedFusionStyle {
        expected: FusionStyleKind,
        actual: FusionStyleKind,
    },
    UnsupportedBraidingStyle {
        expected: &'static str,
        actual: BraidingStyleKind,
    },
    UnsupportedSectorBraid {
        left: SectorId,
        right: SectorId,
        style: BraidingStyleKind,
    },
    InvalidSector {
        sector: SectorId,
    },
    SectorMismatch {
        expected: SectorId,
        actual: SectorId,
    },
    FusionRuleMismatch {
        expected: RuleIdentity,
        actual: RuleIdentity,
    },
    MissingFusionRuleIdentity,
    /// A per-sector leg degeneracy disagrees with another authoritative
    /// source (the paired leg of a composition, or a fusion-tree degeneracy
    /// shape validated against its leg).
    LegDegeneracyMismatch {
        sector: SectorId,
        expected: usize,
        actual: usize,
    },
    FusionChannelCount {
        left: SectorId,
        right: SectorId,
        count: usize,
    },
    MalformedFusionTree {
        message: &'static str,
    },
    BlockCountMismatch {
        expected: usize,
        actual: usize,
    },
    BlockIndexOutOfBounds {
        index: usize,
        count: usize,
    },
    DuplicateBlockKey {
        key: BlockKey,
    },
    MissingBlockKey {
        key: BlockKey,
    },
    MissingFusionSpace,
    /// A bounded fusion table (SU(3) dim<=27, Stage B3b) cannot represent the
    /// requested space/sector exactly. Carries the full human-readable
    /// diagnosis; block dimensions are either exact or this error — never
    /// silently truncated.
    FusionOutsideTable {
        message: String,
    },
    ElementCountOverflow,
    OffsetOverflow {
        value: usize,
    },
    StrideOverflow {
        value: usize,
    },
    OutOfBounds,
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RankMismatch { shape, strides } => {
                write!(
                    f,
                    "rank mismatch: shape rank {shape}, strides rank {strides}"
                )
            }
            Self::StructureRankMismatch { expected, actual } => {
                write!(
                    f,
                    "block structure rank mismatch: expected {expected}, got {actual}"
                )
            }
            Self::FusionSpaceSplitMismatch {
                expected_nout,
                expected_nin,
                actual_nout,
                actual_nin,
            } => {
                write!(
                    f,
                    "fusion-space split mismatch: hom space is {expected_nout} <- {expected_nin}, dynamic split is {actual_nout} <- {actual_nin}"
                )
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::InvalidBraidIndex { index, rank } => {
                write!(
                    f,
                    "cannot braid adjacent fusion-tree outputs at index {index} for rank {rank}"
                )
            }
            Self::InvalidPermutation { permutation, rank } => {
                write!(f, "invalid permutation {permutation:?} for rank {rank}")
            }
            Self::UnsupportedFusionStyle { expected, actual } => {
                write!(
                    f,
                    "unsupported fusion style {actual:?}; expected {expected:?}"
                )
            }
            Self::UnsupportedBraidingStyle { expected, actual } => {
                write!(
                    f,
                    "unsupported braiding style {actual:?}; expected {expected}"
                )
            }
            Self::UnsupportedSectorBraid { left, right, style } => {
                write!(
                    f,
                    "cannot braid non-unit sectors {left:?} and {right:?} with braiding style {style:?}"
                )
            }
            Self::InvalidSector { sector } => write!(f, "invalid sector {sector:?}"),
            Self::SectorMismatch { expected, actual } => {
                write!(f, "sector mismatch: expected {expected:?}, got {actual:?}")
            }
            Self::FusionRuleMismatch { expected, actual } => {
                write!(f, "fusion rule mismatch: expected {expected:?}, got {actual:?}")
            }
            Self::MissingFusionRuleIdentity => write!(f, "fusion space has no bound rule identity"),
            Self::LegDegeneracyMismatch {
                sector,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "leg degeneracy mismatch for sector {sector:?}: expected {expected}, got {actual}"
                )
            }
            Self::FusionChannelCount { left, right, count } => {
                write!(
                    f,
                    "expected one fusion channel for {left:?} x {right:?}, got {count}"
                )
            }
            Self::MalformedFusionTree { message } => {
                write!(f, "malformed fusion tree: {message}")
            }
            Self::BlockCountMismatch { expected, actual } => {
                write!(f, "block count mismatch: expected {expected}, got {actual}")
            }
            Self::BlockIndexOutOfBounds { index, count } => {
                write!(f, "block index {index} is out of bounds for {count} blocks")
            }
            Self::DuplicateBlockKey { key } => {
                write!(f, "duplicate block key {key:?}")
            }
            Self::MissingBlockKey { key } => {
                write!(f, "missing matching block for key {key:?}")
            }
            Self::MissingFusionSpace => write!(f, "tensor does not carry a fusion-tree space"),
            Self::FusionOutsideTable { message } => write!(f, "{message}"),
            Self::ElementCountOverflow => write!(f, "block element count overflow"),
            Self::OffsetOverflow { value } => {
                write!(f, "block offset {value} overflows addressable layout")
            }
            Self::StrideOverflow { value } => {
                write!(f, "block stride {value} overflows addressable layout")
            }
            Self::OutOfBounds => write!(f, "block view accesses outside the buffer"),
        }
    }
}

impl std::error::Error for CoreError {}

pub fn validate_layout(layout: BlockLayout<'_>) -> Result<(), CoreError> {
    if layout.shape.len() != layout.strides.len() {
        return Err(CoreError::RankMismatch {
            shape: layout.shape.len(),
            strides: layout.strides.len(),
        });
    }
    if layout.is_empty() {
        return if layout.offset <= layout.len {
            Ok(())
        } else {
            Err(CoreError::OutOfBounds)
        };
    }
    if layout.offset >= layout.len {
        return Err(CoreError::OutOfBounds);
    }
    let max_delta = max_offset_delta(layout.shape, layout.strides)?;
    let last = layout
        .offset
        .checked_add(max_delta)
        .ok_or(CoreError::OffsetOverflow {
            value: layout.offset,
        })?;
    if last < layout.len {
        Ok(())
    } else {
        Err(CoreError::OutOfBounds)
    }
}

fn max_offset_delta(shape: &[usize], strides: &[usize]) -> Result<usize, CoreError> {
    shape
        .iter()
        .zip(strides)
        .try_fold(0usize, |acc, (&dim, &stride)| {
            let steps = dim.saturating_sub(1);
            let delta = steps
                .checked_mul(stride)
                .ok_or(CoreError::StrideOverflow { value: stride })?;
            acc.checked_add(delta)
                .ok_or(CoreError::ElementCountOverflow)
        })
}

fn storage_end_exclusive(
    shape: &[usize],
    strides: &[usize],
    offset: usize,
) -> Result<usize, CoreError> {
    if shape.len() != strides.len() {
        return Err(CoreError::RankMismatch {
            shape: shape.len(),
            strides: strides.len(),
        });
    }
    if shape.iter().any(|&dim| dim == 0) {
        return Ok(offset);
    }
    let max_delta = max_offset_delta(shape, strides)?;
    offset
        .checked_add(max_delta)
        .and_then(|last| last.checked_add(1))
        .ok_or(CoreError::OffsetOverflow { value: offset })
}

fn checked_product(dims: &[usize]) -> Result<usize, CoreError> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or(CoreError::ElementCountOverflow)
    })
}

fn column_major_strides(shape: &[usize]) -> Result<Vec<usize>, CoreError> {
    let mut strides = vec![1usize; shape.len()];
    for index in 1..shape.len() {
        strides[index] = strides[index - 1]
            .checked_mul(shape[index - 1])
            .ok_or(CoreError::ElementCountOverflow)?;
    }
    Ok(strides)
}
